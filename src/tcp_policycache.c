#define pr_fmt(fmt) "[spine]: " fmt

#include <linux/math64.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <net/tcp.h>
#include <linux/random.h>

#include "lib/spine.h"
#include "spine_nl.h"
#include "tcp_spine.h"

/* all parameters are devided by 1024 */
#define POLICYCACHE_SCALE 1000
#define CWND_GAIN 2000
#define POLICYCACHE_ACTION_SLOW_START 1100
#define POLICYCACHE_ACTION_INCREASE 1025
#define POLICYCACHE_ACTION_DECREASE 976
// #define POLICYCACHE_ACTION_INCREASE_MINOR 1010
// #define POLICYCACHE_ACTION_DECREASE_MINOR 990
#define POLICYCACHE_ACTION_RTT_UPPERBOUND 1100
#define POLICYCACHE_ACTION_RTT_LOWERBOUND 905

#define POLICYCACHE_PARAM_NUM 1

#define POLICYCACHE_IGNORE_PACKETS 5

#define POLICYCACHE_INTERVALS 100
#define MONITOR_INTERVAL 30000
#define POLICYCACHE_RATE_MIN 1024u

// Add THR_UNIT_DEEPCC definition
#define THR_SCALE_DEEPCC 24
#define THR_UNIT_DEEPCC (1 << THR_SCALE_DEEPCC)

/* pcc parameters */
/* Probing changes rate by 5% up and down of current rate. */
enum PCC_STATE {
	PCC_MOVING,
	PCC_PROBING
};

#define PCC_PROBING_EPS 25
#define PCC_PROBING_EPS_PART 1000

#define PCC_MIN_RATE_PACKETS_PER_RTT 2
#define PCC_INTERVAL_PER_PACKET 50
#define PCC_ALPHA 100

#define PCC_GRAD_STEP_SIZE 25
#define PCC_MAX_SWING_BUFFER 2

#define PCC_LAT_INFL_FILTER 30

/* Rates must differ by at least 2% or gradients are very noisy. */
#define PCC_MIN_RATE_DIFF_RATIO_FOR_GRAD 20

#define PCC_MIN_CHANGE_BOUND 100
#define PCC_CHANGE_BOUND_STEP 70
#define PCC_MIN_AMP 2
/* ---- */

extern struct spine_datapath *kernel_datapath;
extern struct timespec64 tzero;
static int id = 0;
struct policycache_interval {
	u64 rate; /* sending rate of this interval, bytes/sec */
	u64 cwnd;

	u64 recv_start; /* timestamps for when interval was waiting for acks */
	u64 recv_end;

	u64 send_start; /* timestamps for when interval data was being sent */
	u64 send_end;

	u64 start_rtt; /* smoothed RTT at start and end of this interval */
	u64 end_rtt;

	u32 recv_id_when_sent;

	u32 packets_sent_base; /* packets sent when this interval started */
	u32 packets_ended; /* packets sent when this interval ended */

	enum PCC_STATE pcc_state; /* state of pcc at the end of this interval */
	s64 utility; /* observed utility of this interval */	
	u32 lost; /* packets sent during this interval that were lost */
	u32 delivered; /* packets sent during this interval that were delivered */

	u64 avg_throughput;
	u64 thr_cnt;
};

/* TCP POLICYCACHE Parameters */
struct policycache_data {
	int cnt; /*  cwnd change */
	bool in_recovery;
	bool is_probe;
	u32 r_cwnd; /* cwnd in loss or recovery */

	u8 slow_start_passed;

	/* policycache parameters */
	struct policycache_interval *intervals; /* containts stats for 1 RTT */

	int send_index; /* index of interval currently being sent */
	int receive_index; /* index of interval currently receiving acks */

	u64 rate;        // current sending rate
	u64 ready_cwnd; // cwnd updated by RL model, used in the next MI

	u64 cwnd;

	u32 lost_base; /* previously lost packets */
	u32 delivered_base; /* previously delivered packets */

	u32 packets_counted; /* packets received or loss confirmed*/

	/* CA state on previous ACK */
	u32 prev_ca_state : 3;
	/* prior cwnd upon entering loss recovery */
	u32 prior_cwnd;

	bool first_circle;


	u16 last_learned_id;
	u16 last_learned_direction;

	int id;
	/* communication */
	struct spine_connection *conn;

	/* others */
	u32 double_counted;
};

/*****************
 * Util functions *
 * ************/

static u32 get_next_index(u32 index)
{
	if (index < POLICYCACHE_INTERVALS - 1)
		return index + 1;
	return 0;
}

static u32 get_previous_index(u32 index, u32 step)
{
	if (index > step - 1)
		return index - step;
	else
		return POLICYCACHE_INTERVALS - (step - index);
}

/*********************
 * Getters / Setters *
 * ******************/
static u32 policycache_get_rtt(struct tcp_sock *tp)
{
	/* Get initial RTT - as measured by SYN -> SYN-ACK.
	 * If information does not exist - use 1ms as a "LAN RTT".
	 * (originally from BBR).
	 */
	if (tp->srtt_us) {
		return max(tp->srtt_us >> 3, 1U);
	} else {
		return USEC_PER_MSEC;
	}
}

/* Calculate the graident of utility w.r.t. sending rate, but only if the rates
 * are far enough apart for the measurment to have low noise.
 */
static s64 pcc_calc_util_grad(s64 rate_1, s64 util_1, s64 rate_2, s64 util_2) {
	if (rate_1 == rate_2)
		return 0;
	
	s64 rate_diff = rate_2 - rate_1;
	s64 util_diff = util_2 - util_1;

	if ((rate_diff > 0 && util_diff > 0) || (rate_diff < 0 && util_diff < 0))
		return 1;
	else if ((rate_diff > 0 && util_diff < 0) || (rate_diff < 0 && util_diff > 0))
		return -1;
	return 0;
}

static s64 generate_explorative_rate(u64 rate, bool increase)
{
	if (increase) {
		return rate + (rate * PCC_PROBING_EPS) / PCC_PROBING_EPS_PART + 1;
	} else {
		return rate - (rate * PCC_PROBING_EPS) / PCC_PROBING_EPS_PART - 1;
	}
}

// get cwnd based on the rate
static u64 policycache_get_cwnd(struct sock *sk, u64 rate)
{
	struct tcp_sock *tp = tcp_sk(sk);
	u64 cwnd = rate;
	u32 rtt = policycache_get_rtt(tp);
	cwnd *= rtt;
	cwnd /= tp->mss_cache;

	cwnd /= USEC_PER_SEC;
	cwnd *= CWND_GAIN;
	cwnd /= POLICYCACHE_SCALE;
	cwnd = max(4ULL, cwnd);
	cwnd = min((u32)cwnd, tp->snd_cwnd_clamp); /* apply cap */
	return cwnd;
}


u64 policycache_calculate_cwnd_for_probe(struct sock *sk, struct policycache_data *policycache)
{
	int last_received_id, last_last_received_id;
	u64 cwnd, last_cwnd;
	s64 new_cwnd;
	char rand;
	bool is_explorative = false;
	s64 grad;

	if (policycache->first_circle && policycache->receive_index < 2) {
		return policycache->ready_cwnd;
	}

	// get the last two intervals 
	last_received_id = get_previous_index(policycache->receive_index, 1u);
	last_last_received_id = get_previous_index(last_received_id, 1u);
	cwnd = policycache->intervals[last_received_id].cwnd;	
	last_cwnd = policycache->intervals[last_last_received_id].cwnd;
	// printk(KERN_INFO "Info of cwnd: %llu, last_cwnd: %llu\n", cwnd, last_cwnd);
	// printk(KERN_INFO "Info of last_received_id: %d, last_last_received_id: %d\n", last_received_id, last_last_received_id);
	// calculate the utility gradient
	if (cwnd == last_cwnd ){
		is_explorative = true;
	}else{
		grad = pcc_calc_util_grad(
			policycache->intervals[last_received_id].cwnd,
			policycache->intervals[last_received_id].utility,
			policycache->intervals[last_last_received_id].cwnd,
			policycache->intervals[last_last_received_id].utility);
		// printk(KERN_INFO "Info of last interval: cwnd: %llu, rate: %llu, utility: %lld\n",
		// 	policycache->intervals[last_received_id].cwnd, 
		// 	policycache->intervals[last_received_id].rate, 		
		// 	policycache->intervals[last_received_id].utility);
		// printk(KERN_INFO "Info of last two interval: cwnd: %llu, rate: %llu, utility: %lld\n",
		// 	policycache->intervals[last_last_received_id].cwnd, 
		// 	policycache->intervals[last_last_received_id].rate, 
		// 	policycache->intervals[last_last_received_id].utility);	
		// printk(KERN_INFO "Info of grad: %lld\n", grad);
	}

	if (grad == 0) {
		is_explorative = true;
	}

	if (is_explorative) {
		// return a random change on the rate 
		get_random_bytes(&rand, 1);
		new_cwnd = generate_explorative_rate(cwnd, rand & 1);
		new_cwnd = max(new_cwnd, 4ULL);
		return new_cwnd;
	}
	if (grad > 0){
		new_cwnd = policycache->cwnd + (policycache->cwnd * PCC_PROBING_EPS) / PCC_PROBING_EPS_PART + 1;
		policycache->last_learned_id = policycache->intervals[last_received_id].recv_id_when_sent;
		policycache->last_learned_direction = 1;
	}else{
		new_cwnd = policycache->cwnd - (policycache->cwnd * PCC_PROBING_EPS) / PCC_PROBING_EPS_PART - 1;
		policycache->last_learned_id = policycache->intervals[last_received_id].recv_id_when_sent;
		policycache->last_learned_direction = 0;
	}
	new_cwnd = max(new_cwnd, 4ULL);
	return new_cwnd;
}

void policycache_calculate_and_set_rate_and_cwnd(struct sock *sk, struct policycache_data *policycache, struct policycache_interval *interval)
{
	u64 new_cwnd, new_rate, explorative_cwnd;
	struct tcp_sock *tp = tcp_sk(sk);
	// printk(KERN_INFO "Old cwnd: %llu, old rate: %llu\n", policycache->cwnd, policycache->rate);
	explorative_cwnd = policycache_calculate_cwnd_for_probe(sk, policycache);
	if (policycache->is_probe){
		new_cwnd = explorative_cwnd;
	}else{
		new_cwnd = policycache->ready_cwnd;
	}

	// Set cwnd from ready_cwnd (cwnd-first)
	new_cwnd = max(4ULL, new_cwnd);
	new_cwnd = min((u32)new_cwnd, tp->snd_cwnd_clamp); /* apply cap */
	interval->cwnd = new_cwnd;
	policycache->cwnd = new_cwnd;
	policycache->ready_cwnd = new_cwnd;
	tp->snd_cwnd = new_cwnd;

	// Now calculate rate from new cwnd and RTT
	u32 rtt = policycache_get_rtt(tp);
	new_rate = new_cwnd * tp->mss_cache;
	new_rate *= USEC_PER_SEC;
	if (rtt > 0)
		do_div(new_rate, rtt);
	else
		new_rate = POLICYCACHE_RATE_MIN;
	new_rate = max(new_rate, POLICYCACHE_RATE_MIN);
	new_rate = min(new_rate, sk->sk_max_pacing_rate);
	interval->rate = new_rate;
	policycache->rate = new_rate;
	sk->sk_pacing_rate = new_rate;
	// printk(KERN_INFO "New interval: cwnd: %llu, rate: %llu\n", new_cwnd, new_rate);
}

bool policycache_valid(struct policycache_data *policycache)
{
	return (policycache && policycache->intervals && policycache->intervals[0].rate);
}

/* Set the pacing rate and cwnd base on the currently-sending interval */
void start_interval(struct sock *sk, struct policycache_data *policycache)
{
	struct policycache_interval *interval = &policycache->intervals[policycache->send_index];
	interval->packets_ended = 0;
	interval->lost = 0;
	interval->delivered = 0;
	interval->packets_sent_base = max(tcp_sk(sk)->data_segs_out, 1U);
	interval->send_start = tcp_sk(sk)->tcp_mstamp;
	interval->avg_throughput = 0;
	interval->thr_cnt = 0;
	interval->recv_id_when_sent = get_previous_index(policycache->receive_index, 1u);
	// pr_info("Start a interval, packets_sent_base: %d, send_start:%llu\n", interval->packets_sent_base, interval->send_start);
	policycache_calculate_and_set_rate_and_cwnd(sk, policycache, interval);
}

/**************************
 * intervals & sample:
 * was started, was ended,
 * find interval per sample
 * ************************/

/* Have we sent all the data we need to for this interval? Must have at least a MONITER_INTERVAL.*/
bool send_interval_ended(struct policycache_interval *interval, struct tcp_sock *tsk,
			 struct policycache_data *policycache)
{
	u64 now = tsk->tcp_mstamp;
	if (now - interval->send_start >= MONITOR_INTERVAL) {
		interval->packets_ended = tsk->data_segs_out;
		// pr_info ("The send interval ended, packets_ended: %d, data_segs_out: %d", interval->packets_ended, tsk->data_segs_out);
		return true;
	} else
		return false;
}

/* Have we accounted for (acked or lost) enough of the packets that we sent to
 * calculate summary statistics?
 */
bool receive_interval_ended(struct policycache_interval *interval, struct tcp_sock *tsk,
			    struct policycache_data *policycache)
{
	// if current time - the rtt of the last packet > send_end time, then the interval is ended.
	// pr_info ("The current time is %llu, the rtt of the last packet is %llu, the send_end is %llu", tsk->tcp_mstamp, tsk->rack.rtt_us , interval->send_end);
	return tsk->tcp_mstamp - tsk->rack.rtt_us > interval->send_end;	
	
	// return interval->packets_ended &&
	//        interval->packets_ended - POLICYCACHE_IGNORE_PACKETS < policycache->packets_counted;
}

/* Start the next interval's sending stage.
 */
void start_next_send_interval(struct sock *sk, struct policycache_data *policycache)
{
	policycache->send_index = get_next_index(policycache->send_index);
	if (policycache->send_index == policycache->receive_index) {
		printk(KERN_INFO "Fail: not enough interval slots.\n");
		return;
	}
	start_interval(sk, policycache);
}

/* Update the receiving time window and the number of packets lost/delivered
 * based on socket statistics.
 */
void policycache_update_interval(struct policycache_interval *interval, struct policycache_data *policycache,
			 struct sock *sk, const struct rate_sample *rs)
{
	interval->recv_end = tcp_sk(sk)->tcp_mstamp;
	interval->end_rtt = tcp_sk(sk)->srtt_us >> 3;
	interval->lost += tcp_sk(sk)->lost - policycache->lost_base;
	interval->delivered += tcp_sk(sk)->delivered - policycache->delivered_base;
	
	if (rs->delivered < 0 || rs->interval_us <= 0)
		return; /* Not a valid observation */
	
	u64 bw = (u64)rs->delivered * THR_UNIT_DEEPCC;
	do_div(bw, rs->interval_us);
	// printk(KERN_INFO "The bw is %llu, the interval_us is %llu, the delivered is %llu\n", bw, rs->interval_us, rs->delivered);
	interval->avg_throughput += bw;
	interval->thr_cnt++;
}

#define VIVACE_LATENCY_COEFFICIENT 900  // scaled by 1000
#define VIVACE_LOSS_COEFFICIENT 11     // scaled by 1000
#define VIVACE_SCALE 1000              // scaling factor for fixed-point math

static void pcc_calc_utility_vivace_latency(struct policycache_data *policycache,
	struct policycache_interval *interval, struct sock *sk) {
s64 loss_ratio, delivered, lost, mss, rate, throughput, util;
	s64 lat_infl = 0;
    s64 rtt_diff;
    s64 rtt_diff_thresh = 0;
	s64 send_dur = interval->send_end - interval->send_start;
	s64 recv_dur = interval->recv_end - interval->recv_start;

	lost = interval->lost;
	delivered = interval->delivered;
	mss = tcp_sk(sk)->mss_cache;
	rate = interval->rate;
	throughput = 0;
	if (recv_dur > 0)
		throughput = (USEC_PER_SEC * delivered * mss) / recv_dur;
	if (delivered == 0) {
        printk(KERN_INFO "No packets delivered\n");
		//interval->utility = S64_MIN;
		interval->utility = 0;
		return;
	}

	rtt_diff = interval->end_rtt - interval->start_rtt;
    if (throughput > 0)
	    rtt_diff_thresh = (2 * USEC_PER_SEC * mss) / throughput;
	if (send_dur > 0)
		lat_infl = (POLICYCACHE_SCALE * rtt_diff) / send_dur;
	
	// printk(KERN_INFO
	// 	"%d ucalc: lat (%lld->%lld) lat_infl %lld\n",
		//  policycache->id, interval->start_rtt / USEC_PER_MSEC, interval->end_rtt / USEC_PER_MSEC,
		//  lat_infl);

	if (rtt_diff < rtt_diff_thresh && rtt_diff > -1 * rtt_diff_thresh)
		lat_infl = 0;

	if (lat_infl < PCC_LAT_INFL_FILTER && lat_infl > -1 * PCC_LAT_INFL_FILTER)
		lat_infl = 0;

	/* loss rate = lost packets / all packets counted*/
	loss_ratio = (lost * POLICYCACHE_SCALE) / (lost + delivered);

	util = /* int_sqrt((u64)rate)*/ rate - (rate * (900 * lat_infl + 11 * loss_ratio)) / POLICYCACHE_SCALE;

	// printk(KERN_INFO
	// 	"%d ucalc: rate %lld sent %u delv %lld lost %lld lat (%lld->%lld) util %lld rate %lld thpt %lld\n",
		//  policycache->id, rate, interval->packets_ended - interval->packets_sent_base,
		//  delivered, lost, interval->start_rtt / USEC_PER_MSEC, interval->end_rtt / USEC_PER_MSEC, util, rate, throughput);
	interval->utility = util;
}

/* Updates the POLICYCACHE model */
void policycache_process(struct sock *sk, const struct rate_sample *rs)
{
	struct policycache_data *policycache = inet_csk_ca(sk);
	struct tcp_sock *tsk = tcp_sk(sk);
	struct policycache_interval *interval;
	int index;
	u32 before;

	if (!policycache_valid(policycache))
		return;
	// policycache_update_pacing_rate(sk);
	/* update send intervals */
	interval = &policycache->intervals[policycache->send_index];
	if (send_interval_ended(interval, tsk, policycache)) {
		// pr_info("sending inverval ended, start the next send at time %llu.", tsk->tcp_mstamp);
		interval->send_end = tcp_sk(sk)->tcp_mstamp;
		start_next_send_interval(sk, policycache);
	}
	/* update recv intervals */
	index = policycache->receive_index;
	interval = &policycache->intervals[index];
	before = policycache->packets_counted;
	policycache->packets_counted = tsk->delivered + tsk->lost -
				policycache->double_counted;
	if (receive_interval_ended(interval, tsk, policycache)) {
		// pr_info("recving inverval ended packet sent base: %d, packets_ended: %d, packets_counted: %d, double_counted: %d", interval->packets_sent_base, interval->packets_ended, policycache->packets_counted, policycache->double_counted);
		// pr_info("data_segs_in: %d, data_segs_out: %d, delivered: %d, lost: %d", tsk->data_segs_in, tsk->data_segs_out, tsk->delivered, tsk->lost);
		policycache_update_interval(interval, policycache, sk, rs);
		pcc_calc_utility_vivace_latency(policycache, interval, sk);
		// update the receive index
		policycache->receive_index = get_next_index(index);
		interval = &policycache->intervals[policycache->receive_index];
		interval->recv_start = tcp_sk(sk)->tcp_mstamp;
		interval->start_rtt = tcp_sk(sk)->srtt_us >> 3;
		if (policycache->receive_index == 0)
			policycache->first_circle = false;
	}else{
		// pr_info("update %d-th recv inverval. the packet count is %d, the double_counted is %d", index, policycache->packets_counted, policycache->double_counted);
		policycache_update_interval(interval, policycache, sk, rs);
	}
}

/** 
 * Spine call this to push updated parameters.
 * The state features we need:
 *    rate: for the RL agent to calculate the next rate.
 *    thr_gradient: (thr_t - thr_{t-1})/thr_{t-1}
 *    rtt_gradient: (RTT_t - RTT_{t-1})/MI
 *    loss_gradient: (1-loss...)
 *    rate_gradient: rate_t/rate_{t-1}
 * 
 * The state the kernel can provide as integers:
 *     delivered, last_delivered, lost, last_loss, rate, last_rate, RTT diff, 
 *
 * ps: For now request_index is not used, just fetch the lastest MI.
 */

u64 get_gap_between_two_intervals(struct policycache_data *policycache, u32 one_id, u32 another_id	) {
	if (another_id == 0) {
		return 0;
	}
	s64 gap = (s64)one_id - (s64)another_id;
	if (gap < 0) {
		gap = gap + POLICYCACHE_INTERVALS;
	}
	return gap;
}

void policycache_fetch_measurements(struct spine_connection *conn,
				   u64 *measurements, u8 *num_fields,
				   u32 request_index)
{
	struct sock *sk;
	get_sock_from_spine(&sk, conn);
	struct tcp_sock *tp = tcp_sk(sk);
	struct policycache_data *policycache = inet_csk_ca(sk);
	*num_fields = 18;
	if (policycache->first_circle && policycache->receive_index < 2) {
		measurements[0] = 0;
		measurements[1] = 0;
		measurements[2] = 0;
		measurements[3] = 0;
		measurements[4] = 0;
		measurements[5] = 0;
		measurements[6] = 0;
		measurements[7] = 0;
		measurements[8] = 0;
		measurements[9] = 0;
		measurements[10] = 0;
		measurements[11] = 0;
		measurements[12] = 0;
		measurements[13] = 0;
		measurements[14] = 0;
		measurements[15] = 0;
		measurements[16] = 0;
		measurements[17] = 0;
		return;
	}
	int last_received_id = get_previous_index(policycache->receive_index, 1u);
	int last_last_received_id = get_previous_index(last_received_id, 1u);
	// policycache->last_used_cwnd = policycache->intervals[last_received_id].cwnd;

	//pr_info("For the last interval: rate: %llu, lost: %llu; delivered: %llu; start_Rtt:%llu, end_rtt:%llu. send_start:%llu, send_end:%llu, recv_start:%llu, recv_end:%llu ", 
					// policycache->intervals[last_received_id].rate,
					// policycache->intervals[last_received_id].lost,
					// policycache->intervals[last_received_id].delivered,
					// policycache->intervals[last_received_id].start_rtt,
					// policycache->intervals[last_received_id].end_rtt,
					// policycache->intervals[last_received_id].send_start,
					// policycache->intervals[last_received_id].send_end,
					// policycache->intervals[last_received_id].recv_start,
					// policycache->intervals[last_received_id].recv_end);
	measurements[0] = policycache->intervals[last_received_id].delivered;
	measurements[1] = policycache->intervals[last_last_received_id].delivered;
	measurements[2] = policycache->intervals[last_received_id].lost;
	measurements[3] = policycache->intervals[last_last_received_id].lost;
	measurements[4] = policycache->intervals[last_received_id].packets_ended - policycache->intervals[last_received_id].packets_sent_base;
	measurements[5] = policycache->intervals[last_last_received_id].packets_ended -  policycache->intervals[last_last_received_id].packets_sent_base; 
	measurements[6] = policycache->intervals[last_received_id].end_rtt;
	measurements[7]	= policycache->intervals[last_received_id].start_rtt;
	// // output rece and send start and end times
	// pr_info ("The last interval: send_start: %llu, send_end: %llu, recv_start: %llu, recv_end: %llu", policycache->intervals[last_received_id].send_start, policycache->intervals[last_received_id].send_end, policycache->intervals[last_received_id].recv_start, policycache->intervals[last_received_id].recv_end);
	// // output their differences
	// pr_info ("The last interval: send_diff: %llu, recv_diff: %llu", policycache->intervals[last_received_id].send_end - policycache->intervals[last_received_id].send_start, policycache->intervals[last_received_id].recv_end - policycache->intervals[last_received_id].recv_start);
	// pr_info ("The difference between send_end and recev_start: %llu", policycache->intervals[last_received_id].send_end - policycache->intervals[last_received_id].recv_start);
	// pr_info ( "The difference between send_end and recev_end: %llu", policycache->intervals[last_received_id].recv_end - policycache->intervals[last_received_id].send_end);

	measurements[8] = policycache->intervals[last_received_id].recv_end -
		    policycache->intervals[last_received_id].recv_start;	
	measurements[9] = policycache->intervals[last_last_received_id].recv_end-
		    policycache->intervals[last_last_received_id].recv_start;	
	measurements[10] = policycache->intervals[last_received_id].send_end -
		    policycache->intervals[last_received_id].send_start;	
	measurements[11] = policycache->intervals[last_last_received_id].send_end-
		    policycache->intervals[last_last_received_id].send_start;	
	// cwnd
	measurements[12] = policycache->intervals[last_received_id].cwnd;
	measurements[13] = policycache->intervals[last_last_received_id].cwnd;
	// current cwnd
	measurements[14] = policycache->cwnd;
	// Convert throughput back to real value by dividing by THR_UNIT_DEEPCC
	if (policycache->intervals[last_received_id].thr_cnt > 0) {
		measurements[15] = (policycache->intervals[last_received_id].avg_throughput / (policycache->intervals[last_received_id].thr_cnt)) * tp->mss_cache * USEC_PER_SEC / THR_UNIT_DEEPCC;
	} else {
		measurements[15] = 0;
	}
	measurements[16] = get_gap_between_two_intervals(policycache, last_received_id, policycache->last_learned_id);
	measurements[17] = policycache->last_learned_direction;
	policycache->last_learned_direction = 2; // userspace can use 2 to indicate duplicated samples
}

/**
 * Spine call this to fetch updated parameters.
 */
void policycache_set_params(struct spine_connection *conn, u64 *params, u8 num_fields)
{
	struct sock *sk;
	int recent_rtt;
	get_sock_from_spine(&sk, conn);
	struct policycache_data *ca = inet_csk_ca(sk);

	if (conn == NULL || params == NULL) {
		pr_info("%s:conn/params is NULL\n", __FUNCTION__);
		return;
	}

	if (unlikely(conn->flow_info.alg != SPINE_POLICYCACHE) ||
	    unlikely(num_fields != POLICYCACHE_PARAM_NUM)) {
		pr_info("Unknown internal congestion control algorithm, do nothing. %d",
			num_fields);
		return;
	}

	// printk(KERN_INFO "The cwnd is %llu\n", ca->cwnd);
	// printk(KERN_INFO "The params[0] is %llu\n", params[0]);

	// 0 is for probe 
	if (params[0] == 0) {
		ca->is_probe = true;
	}else{
		ca->is_probe = false;
	}
	if (params[0] > POLICYCACHE_SCALE) {
		ca->ready_cwnd = ca->cwnd * params[0] / POLICYCACHE_SCALE + 1;
	} else if (params[0] < POLICYCACHE_SCALE) {
		ca->ready_cwnd = ca->cwnd * params[0] / POLICYCACHE_SCALE;
	} else {
		ca->ready_cwnd = ca->cwnd;
	}
	// printk(KERN_INFO "The ready_cwnd is %llu\n", ca->ready_cwnd);
}


static void policycache_release(struct sock *sk)
{
	struct policycache_data *ca = inet_csk_ca(sk);
	if (ca->conn != NULL) {
		pr_info("freeing connection %d", ca->conn->index);
		spine_connection_free(kernel_datapath, ca->conn->index);
	} else {
		pr_info("already freed");
	}
	id--;
	kfree(ca->intervals);
}

static inline void policycache_reset(struct policycache_data *ca)
{
	ca->cnt = 0;
	ca->prev_ca_state = TCP_CA_Open;
	ca->in_recovery = false;
	ca->prior_cwnd = 0;
	ca->r_cwnd = 0;
	ca->slow_start_passed = 0;
}

static void policycache_init(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct policycache_data *ca = inet_csk_ca(sk);
	policycache_reset(ca);

	ca->intervals = kzalloc(sizeof(struct policycache_interval) * POLICYCACHE_INTERVALS,
				GFP_KERNEL);
	if (!ca->intervals) {
		printk(KERN_INFO "init fails\n");
		return;
	}

	id++;
	ca->id = id;
	tp->snd_cwnd = 64;//64; // init value
	ca->cwnd = tp->snd_cwnd;
	ca->ready_cwnd = tp->snd_cwnd;
	ca->is_probe = true;

	ca->rate = POLICYCACHE_RATE_MIN * 512;
	// ca->ready_rate = POLICYCACHE_RATE_MIN * 512;
	ca->send_index = 0;
	ca->receive_index = 0;
	ca->intervals[0].utility = S64_MIN;
	ca->first_circle = true;
	ca->double_counted = 0;
	ca->last_learned_id = 0;
	ca->last_learned_direction = 2;

	start_interval(sk, ca);

	/* create spine flow and register parameters */
	struct spine_datapath_info dp_info = {
		.init_cwnd = tp->snd_cwnd * tp->mss_cache,
		.mss = tp->mss_cache,
		.src_ip = tp->inet_conn.icsk_inet.inet_saddr,
		.src_port = tp->inet_conn.icsk_inet.inet_sport,
		.dst_ip = tp->inet_conn.icsk_inet.inet_daddr,
		.dst_port = tp->inet_conn.icsk_inet.inet_dport,
		.congAlg = "policycache",
		.alg = SPINE_POLICYCACHE,
	};
	// pr_info("New spine flow, from: %u:%u to %u:%u", dp_info.src_ip,
	// 	dp_info.src_port, dp_info.dst_ip, dp_info.dst_port);
	ca->conn =
		spine_connection_start(kernel_datapath, (void *)sk, &dp_info);
	if (ca->conn == NULL) {
		pr_info("start connection failed\n");
	} else {
		pr_info("starting spine connection %d", ca->conn->index);
	}

	// if no ecn support
	if (!(tp->ecn_flags & TCP_ECN_OK)) {
		INET_ECN_dontxmit(sk);
	}

	cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);
}

static void policycache_cong_avoid(struct sock *sk, u32 ack, u32 acked)
{
}

static u32 policycache_ssthresh(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	// we want RL to take more efficient control
	struct policycache_data *ca = inet_csk_ca(sk);
	u64 cwnd;
	struct policycache_interval *interval;
	// if (!ca->slow_start_passed){
	//  	ca->slow_start_passed = 1;
	// }
	// 	tp->snd_cwnd = tp->snd_cwnd * 717 / 1000;
	// 	interval = &ca->intervals[ca->send_index];
	// 	policycache_update_pacing_rate(sk, interval);
	// 	// ca->last_used_cwnd = cwnd;
	// 	cwnd = tp->snd_cwnd;
	// 	ca->intervals[0].cwnd = cwnd;
	// 	ca->ready_cwnd = cwnd;
	// 	ca->cwnd = cwnd;
	// }
	ca->prior_cwnd = tp->snd_cwnd;
	return max(tp->snd_cwnd, 32U);
}

static void policycache_set_state(struct sock *sk, u8 new_state)
{	
	struct policycache_data *policycache = inet_csk_ca(sk);
	struct tcp_sock *tsk = tcp_sk(sk);

	s32 double_counted;

	if (!policycache_valid(policycache))
		return;

	if (policycache->prev_ca_state = TCP_CA_Loss && new_state != TCP_CA_Loss) {
		double_counted = tcp_sk(sk)->delivered + tcp_sk(sk)->lost+
			tcp_packets_in_flight(tcp_sk(sk));
		double_counted -= tcp_sk(sk)->data_segs_out;
		double_counted -= policycache->double_counted;
		policycache->double_counted+= double_counted;
		printk(KERN_INFO "%d loss ended: double_counted %d\n", policycache->id, double_counted);
		policycache->prev_ca_state = new_state;
	}
	else if (policycache->prev_ca_state != TCP_CA_Loss && new_state	== TCP_CA_Loss) {
		// printk(KERN_INFO "%d loss: started\n", pcc->id);
		policycache->prev_ca_state = new_state;
	}
}


static void policycache_pkt_acked(struct sock *sk, const struct ack_sample *sample)
{
}

static u32 policycache_undo_cwnd(struct sock *sk)
{
	return tcp_sk(sk)->snd_cwnd;
}

static void slow_set_cwnd(struct sock *sk, u32 acked)
{
	// do_div(change, POLICYCACHE_SCALE);
	struct tcp_sock *tp = tcp_sk(sk);
	struct policycache_data *ca = inet_csk_ca(sk);
	u32 cwnd = tp->snd_cwnd;
	int delta = ca->cnt;
	// printk(KERN_INFO "Delta before division: %d.\n", delta);

	delta = delta / POLICYCACHE_SCALE;

	if (delta != 0) {
		ca->cnt -= delta * POLICYCACHE_SCALE;
		// printk(KERN_INFO "[POLICYCACHE] Old CWND %d, New CWND %d.\n", cwnd, cwnd + delta);
		cwnd += delta;
	}
	cwnd = max(4ULL, cwnd);
	cwnd = min((u32)cwnd, tp->snd_cwnd_clamp); /* apply cap */
	tp->snd_cwnd = cwnd;

}

// u32 policycache_slow_start(struct sock *sk, u32 acked)
// {
// 	struct tcp_sock *tp = tcp_sk(sk);
// 	struct policycache_data *ca = inet_csk_ca(sk);
// 	u64 rate;
// 	ca->cnt += acked * 500;
// 	slow_set_cwnd(sk, acked);
// 	policycache_update_pacing_rate(sk);

// 	rate = sk->sk_pacing_rate;
// 	ca->intervals[0].rate = rate;
// 	ca->ready_rate = rate;
// 	ca->rate = rate;
// }

static void policycache_cong_control(struct sock *sk, const struct rate_sample *rs)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct policycache_data *ca = inet_csk_ca(sk);
	struct spine_connection *conn = ca->conn;
	u8 prev_state = ca->prev_ca_state, state = inet_csk(sk)->icsk_ca_state;
	u32 acked = rs->acked_sacked; //rs->delivered;
	int ok = 0;
	// printk(KERN_INFO "[POLICYCACHE] Get into control1.\n");
	// we only do slow start when flow starts
	// if (tcp_in_slow_start(tp) && !ca->slow_start_passed) {
	// 	// printk(KERN_INFO "[POLICYCACHE] acked: %d, delivered %d.\n, ",  rs->acked_sacked, rs->delivered);
	// 	policycache_slow_start(sk, acked);
	// 	goto end;
	// }

	if (prev_state >= TCP_CA_Recovery && state < TCP_CA_Recovery) {
		/* Exiting loss recovery; restore cwnd saved before recovery. */
		tp->snd_cwnd = max(tp->snd_cwnd, ca->prior_cwnd);
	}

	if (rs->delivered < 0 || rs->interval_us < 0) {
		goto end;
	}
	// pr_info("The acked is %d, the delivered is %d, the interval_us is %d", rs->acked_sacked, rs->delivered, rs->interval_us);
	policycache_process(sk, rs);
	// printk(KERN_INFO "[POLICYCACHE] Get into control1.\n");
	// call spine to update parameters if needed
	if (conn != NULL) {
		// if there are staged parameters update, then
		// corressponding params inside ca would be updated
		ok = spine_invoke(conn);
		if (ok < 0) {
			pr_info("fail to call spine_invoke: %d\n", ok);
		}
	}
end:
	ca->lost_base = tp->lost;
	ca->delivered_base = tp->delivered;
}

static struct tcp_congestion_ops policycache __read_mostly = {
	.init = policycache_init,
	.release = policycache_release,
	.ssthresh = policycache_ssthresh,
	// .cong_avoid = policycache_cong_avoid,
	.cong_control = policycache_cong_control,
	.set_state = policycache_set_state,
	.undo_cwnd = policycache_undo_cwnd,
	// .cwnd_event = policycache_cwnd_event,
	.pkts_acked = policycache_pkt_acked,
	.owner = THIS_MODULE,
	.name = "policycache",
};

static int __init policycache_register(void)
{
	int ret;
	BUILD_BUG_ON(sizeof(struct policycache_data) > ICSK_CA_PRIV_SIZE);
	ktime_get_real_ts64(&tzero);

	/* Init spine-related structs inspired by CCP
	 * kernel_datapath
	 * spine connections
	 */
	kernel_datapath = (struct spine_datapath *)kmalloc(
		sizeof(struct spine_datapath), GFP_KERNEL);
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
		(struct spine_connection *)kzalloc(
			sizeof(struct spine_connection) * MAX_ACTIVE_FLOWS,
			GFP_KERNEL);
	if (!kernel_datapath->spine_active_connections) {
		pr_info("could not allocate spine_connections\n");
		return -2;
	}
	kernel_datapath->log = &spine_log;
	kernel_datapath->set_params = &policycache_set_params;
	kernel_datapath->fetch_measurements = &policycache_fetch_measurements;
	kernel_datapath->send_msg = &nl_sendmsg;

	/* Here we need to add a IPC for receiving messages from user space 
	 * RL controller.
	 */
	ret = spine_nl_sk(spine_read_msg);
	if (ret < 0) {
		pr_info("cannot init spine ipc\n");
		return -3;
	}
	pr_info("spine ipc init\n");
	// register current sock in spine datapath
	ret = spine_init(kernel_datapath, 0);
	if (ret < 0) {
		pr_info("fail to init spine datapath\n");
		free_spine_nl_sk();
		return -4;
	}
	pr_info("spine %s init\n", policycache.name);

	return tcp_register_congestion_control(&policycache);
}

static void __exit policycache_unregister(void)
{
	free_spine_nl_sk();
	kfree(kernel_datapath->spine_active_connections);
	kfree(kernel_datapath);
	pr_info("spine exit\n");
	tcp_unregister_congestion_control(&policycache);
}

module_init(policycache_register);
module_exit(policycache_unregister);

MODULE_AUTHOR("Han Tian");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("TCP Policy Cache");
MODULE_VERSION("1.0");
