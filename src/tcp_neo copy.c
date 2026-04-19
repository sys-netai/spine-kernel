#define pr_fmt(fmt) "[spine]: " fmt

#include <linux/math64.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <net/tcp.h>

#include "lib/spine.h"
#include "spine_nl.h"
#include "tcp_spine.h"

/* all parameters are divided by 1000 */
#define NEO_SCALE 1000
#define CWND_GAIN 1200
#define NEO_ACTION_SLOW_START 1100
#define NEO_ACTION_INCREASE 1025
#define NEO_ACTION_DECREASE 976
#define NEO_ACTION_RTT_UPPERBOUND 1100
#define NEO_ACTION_RTT_LOWERBOUND 905

#define NEO_PARAM_NUM 1

/* how many packets near the boundary we tolerate before closing an interval */
#define NEO_IGNORE_PACKETS 5

#define NEO_INTERVALS 100
#define MONITOR_INTERVAL 30000
#define NEO_RATE_MIN 4096u

/* throughput fixed-point scaling */
#define THR_SCALE_DEEPCC 24
#define THR_UNIT_DEEPCC (1 << THR_SCALE_DEEPCC)

extern struct spine_datapath *kernel_datapath;
extern struct timespec64 tzero;

static int id = 0;

struct neo_interval {
	u64 rate; /* sending rate of this interval, bytes/sec */
	u64 cwnd;

	u64 recv_start; /* timestamps for when interval was waiting for acks */
	u64 recv_end;

	u64 send_start; /* timestamps for when interval data was being sent */
	u64 send_end;

	u64 start_rtt; /* smoothed RTT at start and end of this interval */
	u64 end_rtt;

	u32 packets_sent_base; /* packets sent when this interval started */
	u32 packets_ended;     /* packets sent when this interval ended */

	u32 lost;      /* packets sent during this interval that were lost */
	u32 delivered; /* packets sent during this interval that were delivered */

	/*
	 * send_id and receive_id are intentionally kept the same for the same
	 * logical interval, so userspace can align send/receive with one id.
	 */
	u32 send_id;
	u32 receive_id;

	/* throughput tracking */
	u64 avg_throughput;
	u64 thr_cnt;
};

/* TCP NEO parameters */
struct neo_data {
	int cnt; /* cwnd change */
	bool in_recovery;
	u32 r_cwnd; /* cwnd in loss or recovery */

	u8 slow_start_passed;

	/* interval ring */
	struct neo_interval *intervals;

	u32 send_index;    /* index of interval currently being sent */
	u32 receive_index; /* index of interval currently receiving acks */

	u64 cwnd;
	u64 ready_cwnd;

	u32 lost_base;      /* previously lost packets */
	u32 delivered_base; /* previously delivered packets */

	u32 packets_counted; /* delivered + lost - double_counted */

	/* CA state on previous ACK */
	u32 prev_ca_state : 3;
	/* prior cwnd upon entering loss recovery */
	u32 prior_cwnd;

	bool first_circle;

	int id;
	/* communication */
	struct spine_connection *conn;

	/* others */
	u32 double_counted;

	/*
	 * A single logical interval id shared by the send phase and receive phase
	 * of the same ring slot.
	 */
	u32 interval_id_counter;
};

/*****************
 * Util functions *
 *****************/

static u32 get_next_index(u32 index)
{
	if (index < NEO_INTERVALS - 1)
		return index + 1;
	return 0;
}

static u32 get_previous_index(u32 index, u32 step)
{
	if (index > step - 1)
		return index - step;
	else
		return NEO_INTERVALS - (step - index);
}

static void neo_reset_interval(struct neo_interval *interval)
{
	memset(interval, 0, sizeof(*interval));
}

/*********************
 * Getters / Setters *
 *********************/

static u32 neo_get_rtt(struct tcp_sock *tp)
{
	/* Get initial RTT as measured by SYN -> SYN-ACK.
	 * If unavailable, use 1ms as a LAN RTT fallback.
	 */
	if (tp->srtt_us)
		return max(tp->srtt_us >> 3, 1U);
	else
		return USEC_PER_MSEC;
}

static bool neo_valid(struct neo_data *neo)
{
	return neo && neo->intervals;
}

static void neo_calculate_and_set_cwnd(struct sock *sk, struct neo_data *neo,
				       struct neo_interval *interval)
{
	struct tcp_sock *tp = tcp_sk(sk);
	u64 new_cwnd;

	new_cwnd = neo->ready_cwnd;
	new_cwnd = max(4ULL, new_cwnd);
	new_cwnd = min((u32)new_cwnd, tp->snd_cwnd_clamp);

	interval->cwnd = new_cwnd;
	neo->cwnd = new_cwnd;
	neo->ready_cwnd = new_cwnd; /* reuse if no new action arrives */
	tp->snd_cwnd = new_cwnd;
}

static void neo_update_pacing_rate(struct sock *sk)
{
	const struct tcp_sock *tp = tcp_sk(sk);
	u64 rate;

	cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);

	rate = tcp_mss_to_mtu(sk, tp->mss_cache);
	rate *= USEC_PER_SEC;
	rate *= max(tp->snd_cwnd, tp->packets_out);

	if (likely(tp->srtt_us >> 3))
		do_div(rate, tp->srtt_us >> 3);

	WRITE_ONCE(sk->sk_pacing_rate, min_t(u64, rate, sk->sk_max_pacing_rate));
}

static void neo_begin_receive_interval(struct sock *sk, struct neo_data *neo,
				       u32 index)
{
	struct neo_interval *interval = &neo->intervals[index];
	struct tcp_sock *tp = tcp_sk(sk);

	/* Do not initialize receive phase before the send phase has started. */
	if (interval->send_start == 0)
		return;

	/* Only initialize once for this logical interval. */
	if (interval->recv_start != 0)
		return;

	interval->recv_start = tp->tcp_mstamp;
	interval->start_rtt = tp->srtt_us >> 3;
}

/* Set pacing rate and cwnd based on the currently-sending interval */
static void start_interval(struct sock *sk, struct neo_data *neo)
{
	struct neo_interval *interval = &neo->intervals[neo->send_index];

	/* Clear old content before reusing this ring slot. */
	neo_reset_interval(interval);

	/* One logical id shared by send and receive phases of this interval. */
	interval->send_id = ++neo->interval_id_counter;
	interval->receive_id = interval->send_id;

	interval->packets_sent_base = max(tcp_sk(sk)->data_segs_out, 1U);
	interval->send_start = tcp_sk(sk)->tcp_mstamp;

	neo_calculate_and_set_cwnd(sk, neo, interval);
	neo_update_pacing_rate(sk);
}

/**************************
 * intervals & sampling
 **************************/

/* Have we sent all the data we need for this interval? */
static bool send_interval_ended(struct neo_interval *interval,
				struct tcp_sock *tsk,
				struct neo_data *neo)
{
	u64 now = tsk->tcp_mstamp;

	if (interval->send_start == 0)
		return false;

	if (interval->send_end != 0)
		return true;

	if (now - interval->send_start >= MONITOR_INTERVAL) {
		interval->packets_ended = tsk->data_segs_out;
		return true;
	}

	return false;
}

/*
 * Have we accounted for enough of the packets sent in this interval?
 *
 * This version uses packet-accounting based alignment instead of pure time
 * heuristics. That gives much tighter send/receive alignment with the
 * information available from TCP aggregate counters.
 */
static bool receive_interval_ended(struct neo_interval *interval,
				   struct tcp_sock *tsk,
				   struct neo_data *neo)
{
	if (interval->send_end == 0)
		return false;

	if (interval->packets_ended == 0)
		return false;

	if ((s32)(neo->packets_counted + NEO_IGNORE_PACKETS - interval->packets_ended) >= 0) {
		return true;
	}

	u32 rtt = tsk->rack.rtt_us ? tsk->rack.rtt_us : (tsk->srtt_us >> 3);
	rtt = max(rtt, 1000U); // 至少给 1ms 宽限
    if (tsk->tcp_mstamp > interval->send_end + rtt) {
       return true;
    }

    return false;
}

/* Start the next interval's sending stage. */
static void start_next_send_interval(struct sock *sk, struct neo_data *neo)
{
	u32 next = get_next_index(neo->send_index);

	if (next == neo->receive_index) {
		printk(KERN_INFO "Fail: not enough interval slots.\n");
		return;
	}

	neo->send_index = next;
	start_interval(sk, neo);
}

/*
 * Update receiving time window and packet statistics based on socket stats.
 *
 * Must be called before receive_interval_ended() so the boundary sample is
 * included in the interval that is about to close.
 */
static void neo_update_interval(struct neo_interval *interval,
				struct neo_data *neo,
				struct sock *sk,
				const struct rate_sample *rs)
{
	struct tcp_sock *tp = tcp_sk(sk);
	u64 bw;

	if (interval->send_start == 0)
		return;

	if (interval->recv_start == 0)
		neo_begin_receive_interval(sk, neo, neo->receive_index);

	interval->recv_end = tp->tcp_mstamp;
	interval->end_rtt = tp->srtt_us >> 3;
	interval->lost += tp->lost - neo->lost_base;
	interval->delivered += tp->delivered - neo->delivered_base;

	if (rs->delivered < 0 || rs->interval_us <= 0)
		return; /* not a valid throughput sample */

	bw = (u64)rs->delivered * THR_UNIT_DEEPCC;
	do_div(bw, rs->interval_us);
	interval->avg_throughput += bw;
	interval->thr_cnt++;
}

/* Updates the NEO model */
static void neo_process(struct sock *sk, const struct rate_sample *rs)
{
	struct neo_data *neo = inet_csk_ca(sk);
	struct tcp_sock *tsk = tcp_sk(sk);
	struct neo_interval *interval;
	u32 index;

	if (!neo_valid(neo))
		return;

	neo_update_pacing_rate(sk);

	/* update send interval */
	interval = &neo->intervals[neo->send_index];
	if (send_interval_ended(interval, tsk, neo)) {
		if (interval->send_end == 0)
			interval->send_end = tsk->tcp_mstamp;
		start_next_send_interval(sk, neo);
	}

	/* update receive interval */
	index = neo->receive_index;
	interval = &neo->intervals[index];

	neo->packets_counted = tsk->delivered + tsk->lost - neo->double_counted;

	/* Important: update first, then test end. */
	neo_update_interval(interval, neo, sk, rs);

	if (receive_interval_ended(interval, tsk, neo)) {
		neo->receive_index = get_next_index(index);

		if (neo->receive_index == 0)
			neo->first_circle = false;

		/*
		 * The next receive interval corresponds to the next ring slot.
		 * Its send phase should already have started in normal operation.
		 */
		neo_begin_receive_interval(sk, neo, neo->receive_index);
	}
}

/**
 * Spine calls this to fetch updated measurements.
 */
static void neo_fetch_measurements(struct spine_connection *conn,
				   u64 *measurements, u8 *num_fields,
				   u32 request_index)
{
	struct sock *sk;
	struct tcp_sock *tp;
	struct neo_data *neo;
	u32 last_received_idx;
	u32 last_last_received_idx;
	struct neo_interval *send_interval;
	u64 avg_thr;

	get_sock_from_spine(&sk, conn);
	tp = tcp_sk(sk);
	neo = inet_csk_ca(sk);

	*num_fields = 18;

	send_interval = &neo->intervals[neo->send_index];

	if (neo->first_circle && neo->receive_index < 2) {
		memset(measurements, 0, 18 * sizeof(*measurements));
		measurements[16] = send_interval->send_id;

		if (neo->receive_index == 0)
			measurements[17] = 0; /* no completed receive interval yet */
		else
			measurements[17] =
				neo->intervals[get_previous_index(neo->receive_index, 1u)].receive_id;
		return;
	}

	last_received_idx = get_previous_index(neo->receive_index, 1u);
	last_last_received_idx = get_previous_index(last_received_idx, 1u);

	measurements[0] = neo->intervals[last_received_idx].delivered;
	measurements[1] = neo->intervals[last_last_received_idx].delivered;
	measurements[2] = neo->intervals[last_received_idx].lost;
	measurements[3] = neo->intervals[last_last_received_idx].lost;
	measurements[4] = neo->intervals[last_received_idx].packets_ended -
			  neo->intervals[last_received_idx].packets_sent_base;
	measurements[5] = neo->intervals[last_last_received_idx].packets_ended -
			  neo->intervals[last_last_received_idx].packets_sent_base;
	measurements[6] = neo->intervals[last_received_idx].end_rtt;
	measurements[7] = neo->intervals[last_received_idx].start_rtt;
	measurements[8] = neo->intervals[last_received_idx].recv_end -
			  neo->intervals[last_received_idx].recv_start;
	measurements[9] = neo->intervals[last_last_received_idx].recv_end -
			  neo->intervals[last_last_received_idx].recv_start;
	measurements[10] = neo->intervals[last_received_idx].send_end -
			   neo->intervals[last_received_idx].send_start;
	measurements[11] = neo->intervals[last_last_received_idx].send_end -
			   neo->intervals[last_last_received_idx].send_start;
	measurements[12] = neo->intervals[last_received_idx].cwnd;
	measurements[13] = neo->intervals[last_last_received_idx].cwnd;
	measurements[14] = neo->cwnd;

	if (neo->intervals[last_received_idx].thr_cnt > 0) {
		avg_thr = neo->intervals[last_received_idx].avg_throughput /
			  neo->intervals[last_received_idx].thr_cnt;

		if (avg_thr <= U64_MAX / (tp->mss_cache * USEC_PER_SEC / THR_UNIT_DEEPCC))
			measurements[15] =
				avg_thr * tp->mss_cache * USEC_PER_SEC / THR_UNIT_DEEPCC;
		else
			measurements[15] = U64_MAX;
	} else {
		measurements[15] = 0;
	}

	/* current sending interval id */
	measurements[16] = send_interval->send_id;
	/* latest completed receive interval id (same namespace as send_id) */
	measurements[17] = neo->intervals[last_received_idx].receive_id;
}

/**
 * Spine calls this to push updated parameters.
 */
static void neo_set_params(struct spine_connection *conn, u64 *params, u8 num_fields)
{
	struct sock *sk;
	struct neo_data *ca;

	get_sock_from_spine(&sk, conn);
	ca = inet_csk_ca(sk);

	if (conn == NULL || params == NULL) {
		pr_info("%s: conn/params is NULL\n", __func__);
		return;
	}

	if (unlikely(conn->flow_info.alg != SPINE_NEO) ||
	    unlikely(num_fields != NEO_PARAM_NUM)) {
		pr_info("Unknown internal congestion control algorithm, do nothing. %d",
			num_fields);
		return;
	}

	/* continuous multiplicative cwnd control */
	if (params[0] > NEO_SCALE) {
		ca->ready_cwnd = ca->cwnd * params[0] / NEO_SCALE + 1;
	} else if (params[0] < NEO_SCALE) {
		ca->ready_cwnd = ca->cwnd * params[0] / NEO_SCALE;
	} else {
		ca->ready_cwnd = ca->cwnd;
	}
}

static void neo_release(struct sock *sk)
{
	struct neo_data *ca = inet_csk_ca(sk);

	if (ca->conn != NULL) {
		pr_info("freeing connection %d", ca->conn->index);
		spine_connection_free(kernel_datapath, ca->conn->index);
	} else {
		pr_info("already freed");
	}

	id--;
	kfree(ca->intervals);
}

static inline void neo_reset(struct neo_data *ca)
{
	ca->cnt = 0;
	ca->prev_ca_state = TCP_CA_Open;
	ca->in_recovery = false;
	ca->prior_cwnd = 0;
	ca->r_cwnd = 0;
	ca->slow_start_passed = 0;
}

static void neo_init(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct neo_data *ca = inet_csk_ca(sk);

	neo_reset(ca);

	ca->intervals = kzalloc(sizeof(struct neo_interval) * NEO_INTERVALS,
				GFP_KERNEL);
	if (!ca->intervals) {
		printk(KERN_INFO "init fails\n");
		return;
	}

	id++;
	ca->id = id;

	tp->snd_cwnd = 64;
	ca->cwnd = tp->snd_cwnd;
	ca->ready_cwnd = tp->snd_cwnd;

	ca->send_index = 0;
	ca->receive_index = 0;
	ca->first_circle = true;
	ca->double_counted = 0;
	ca->interval_id_counter = 0;

	start_interval(sk, ca);
	neo_begin_receive_interval(sk, ca, ca->receive_index);

	/* create spine flow and register parameters */
	{
		struct spine_datapath_info dp_info = {
			.init_cwnd = tp->snd_cwnd * tp->mss_cache,
			.mss = tp->mss_cache,
			.src_ip = tp->inet_conn.icsk_inet.inet_saddr,
			.src_port = tp->inet_conn.icsk_inet.inet_sport,
			.dst_ip = tp->inet_conn.icsk_inet.inet_daddr,
			.dst_port = tp->inet_conn.icsk_inet.inet_dport,
			.congAlg = "neo",
			.alg = SPINE_NEO,
		};

		ca->conn = spine_connection_start(kernel_datapath, (void *)sk, &dp_info);
		if (ca->conn == NULL)
			pr_info("start connection failed\n");
		else
			pr_info("starting spine connection %d", ca->conn->index);
	}

	/* if no ECN support */
	if (!(tp->ecn_flags & TCP_ECN_OK))
		INET_ECN_dontxmit(sk);

	cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);
}

static u32 neo_ssthresh(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct neo_data *ca = inet_csk_ca(sk);
	u64 cwnd;

	if (!ca->slow_start_passed) {
		ca->slow_start_passed = 1;
		tp->snd_cwnd = tp->snd_cwnd * 717 / 1000;
		neo_update_pacing_rate(sk);

		cwnd = tp->snd_cwnd;
		ca->intervals[0].cwnd = cwnd;
		ca->ready_cwnd = cwnd;
		ca->cwnd = cwnd;
	}

	ca->prior_cwnd = tp->snd_cwnd;
	return max(tp->snd_cwnd, 32U);
}

static void neo_set_state(struct sock *sk, u8 new_state)
{
	struct neo_data *neo = inet_csk_ca(sk);
	struct tcp_sock *tsk = tcp_sk(sk);
	s32 double_counted;

	if (!neo_valid(neo))
		return;

	if (new_state == TCP_CA_Loss) {
		neo->prev_ca_state = TCP_CA_Loss;
		// double_counted = tsk->delivered + tsk->lost +
		// 		 tcp_packets_in_flight(tsk);
		// double_counted -= tsk->data_segs_out;
		// double_counted -= neo->double_counted;
		// neo->double_counted += double_counted;
		neo->double_counted = tsk->delivered + tsk->lost + tcp_packets_in_flight(tsk) - tsk->data_segs_out;
	} else {
		neo->prev_ca_state = new_state;
	}
}

static void neo_pkt_acked(struct sock *sk, const struct ack_sample *sample)
{
}

static u32 neo_undo_cwnd(struct sock *sk)
{
	return tcp_sk(sk)->snd_cwnd;
}

static void neo_cong_control(struct sock *sk, const struct rate_sample *rs)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct neo_data *ca = inet_csk_ca(sk);
	struct spine_connection *conn = ca->conn;
	u8 prev_state = ca->prev_ca_state;
	u8 state = inet_csk(sk)->icsk_ca_state;
	int ok = 0;

	if (prev_state >= TCP_CA_Recovery && state < TCP_CA_Recovery) {
		/* Exiting loss recovery; restore cwnd saved before recovery. */
		tp->snd_cwnd = max(tp->snd_cwnd, ca->prior_cwnd);
	}

	if (rs->delivered < 0 || rs->interval_us < 0)
		goto end;

	neo_process(sk, rs);

	if (conn != NULL) {
		ok = spine_invoke(conn);
		if (ok < 0)
			pr_info("fail to call spine_invoke: %d\n", ok);
	}

end:
	ca->lost_base = tp->lost;
	ca->delivered_base = tp->delivered;
}

static struct tcp_congestion_ops neo __read_mostly = {
	.init		= neo_init,
	.release	= neo_release,
	.ssthresh	= neo_ssthresh,
	.cong_control	= neo_cong_control,
	.set_state	= neo_set_state,
	.undo_cwnd	= neo_undo_cwnd,
	.pkts_acked	= neo_pkt_acked,
	.owner		= THIS_MODULE,
	.name		= "neo",
};

static int __init neo_register(void)
{
	int ret;

	BUILD_BUG_ON(sizeof(struct neo_data) > ICSK_CA_PRIV_SIZE);
	ktime_get_real_ts64(&tzero);

	/* Init spine-related structs inspired by CCP */
	kernel_datapath = kmalloc(sizeof(struct spine_datapath), GFP_KERNEL);
	if (!kernel_datapath) {
		pr_info("could not allocate spine_datapath\n");
		return -1;
	}

	kernel_datapath->now = &spine_now;
	kernel_datapath->since_usecs = &spine_since;
	kernel_datapath->after_usecs = &spine_after;
	kernel_datapath->log = &spine_log;
	kernel_datapath->fto_us = 1000;
	kernel_datapath->max_connections = MAX_ACTIVE_FLOWS;

	kernel_datapath->spine_active_connections =
		kzalloc(sizeof(struct spine_connection) * MAX_ACTIVE_FLOWS,
			GFP_KERNEL);
	if (!kernel_datapath->spine_active_connections) {
		pr_info("could not allocate spine_connections\n");
		kfree(kernel_datapath);
		return -2;
	}

	kernel_datapath->set_params = &neo_set_params;
	kernel_datapath->fetch_measurements = &neo_fetch_measurements;
	kernel_datapath->send_msg = &nl_sendmsg;

	ret = spine_nl_sk(spine_read_msg);
	if (ret < 0) {
		pr_info("cannot init spine ipc\n");
		kfree(kernel_datapath->spine_active_connections);
		kfree(kernel_datapath);
		return -3;
	}
	pr_info("spine ipc init\n");

	ret = spine_init(kernel_datapath, 0);
	if (ret < 0) {
		pr_info("fail to init spine datapath\n");
		free_spine_nl_sk();
		kfree(kernel_datapath->spine_active_connections);
		kfree(kernel_datapath);
		return -4;
	}
	pr_info("spine %s init\n", neo.name);

	return tcp_register_congestion_control(&neo);
}

static void __exit neo_unregister(void)
{
	free_spine_nl_sk();
	kfree(kernel_datapath->spine_active_connections);
	kfree(kernel_datapath);
	pr_info("spine exit\n");
	tcp_unregister_congestion_control(&neo);
}

module_init(neo_register);
module_exit(neo_unregister);

MODULE_AUTHOR("Han Tian");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("TCP Neo (aligned interval version)");
MODULE_VERSION("1.1");
