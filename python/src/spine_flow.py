import os
from socket import socket
import sys
import ipaddress
from typing import Optional
from enum import Enum
from message import *

class ValueType(Enum):
    SOCK_ID = 0
    FLOW_ID = 1

class Flow(object):
    def __init__(self):
        self.sock_id = -1
        # flow info
        self.init_cwnd = 4
        self.mss = 0
        self.src_ip = 0
        self.src_port = 0
        self.dst_ip = 0
        self.dst_port = 0
        # max 64 bytes
        self.congAlg = ""

    def from_create_msg(self, msg: CreateMsg, hdr: SpineMsgHeader):
        self.init_cwnd = msg.init_cwnd
        self.mss = msg.mss
        self.src_ip = msg.src_ip
        self.src_port = msg.src_port
        self.dst_ip = msg.dst_ip
        self.dst_port = msg.dst_port
        # max 64 bytes
        self.congAlg = msg.congAlg

        # key in flow map
        self.sock_id = hdr.sock_id # * self.dst_port 
        return self


class ActiveFlowMap(object):
    def __init__(self):
        # key is sock id (assigned by spine-kernel), value is Flow object
        self.kernel_flows = dict()
        # key is flow id (assigned by Env), value is sock id
        self.flow_id_to_sock_id = dict()
        self.sock_id_to_flow_id = dict()
        # key is (dst_ip, dst_port) tuple, value is flow id
        self.dst_endpoint_to_flow_id = dict()
        self.dst_endpoint_to_sock_id = dict()
         
        self.env_id = None

    def get_all_flow_ids(self):
        return self.flow_id_to_sock_id.keys()

    def try_associate(self, dst_endpoint, id, value_type):
        if value_type == ValueType.SOCK_ID:
            sock_id = id
            # print("try_associate sock id with flow id under dst_endpoint:", dst_endpoint)
            if dst_endpoint in self.dst_endpoint_to_flow_id: # flow id already exists for endpoint
                # first look up flow id from send endpoint
                flow_id = self.dst_endpoint_to_flow_id[dst_endpoint]
                self.flow_id_to_sock_id[flow_id] = sock_id
                self.sock_id_to_flow_id[sock_id] = flow_id
                # print("associate flow id: {} with sock id: {}".format(flow_id, sock_id))
            else:
                # print("no flow id to associate with sock id:", dst_endpoint, sock_id)
                pass
        elif value_type == ValueType.FLOW_ID:
            flow_id = id
            # print("try_associate flow id with sock id:", dst_endpoint)
            if dst_endpoint in self.dst_endpoint_to_sock_id: # sock id already exists for endpoint
                sock_id = self.dst_endpoint_to_sock_id[dst_endpoint]
                self.flow_id_to_sock_id[flow_id] = sock_id
                self.sock_id_to_flow_id[sock_id] = flow_id
                # print("associate flow id: {} with sock id: {}".format(flow_id, sock_id))
            else:
                # print("no sock id to associate with flow id:", dst_endpoint, flow_id)
                pass

    def add_flow_with_sockId(self, flow: Flow):
        dst_endpoint = (flow.dst_ip, flow.dst_port)
        # print("added flow with sockId:", dst_endpoint, flow.sock_id)
        if dst_endpoint not in self.dst_endpoint_to_sock_id:
            self.dst_endpoint_to_sock_id[dst_endpoint] = flow.sock_id
            if flow.sock_id not in self.kernel_flows:
                self.kernel_flows[flow.sock_id] = flow
                log.info(
                    "env: {} add kernel flow: {} with init_cwnd: {},src_ip: {}, src_port: {}, dst_ip: {}, dst_port: {}".format(
                        self.env_id,
                        flow.sock_id,
                        flow.init_cwnd,
                        ipaddress.IPv4Address(flow.src_ip),
                        flow.src_port,
                        ipaddress.IPv4Address(flow.dst_ip),
                        flow.dst_port,
                    )
                )
                self.try_associate(dst_endpoint, flow.sock_id, ValueType.SOCK_ID)
                return True
            else:
                log.warning("flow already exists: {}, but...".format(flow.sock_id))
                self.kernel_flows[flow.sock_id] = flow
                log.info(
                    "env: {} add kernel flow: {} with init_cwnd: {},src_ip: {}, src_port: {}, dst_ip: {}, dst_port: {}".format(
                        self.env_id,
                        flow.sock_id,
                        flow.init_cwnd,
                        ipaddress.IPv4Address(flow.src_ip),
                        flow.src_port,
                        ipaddress.IPv4Address(flow.dst_ip),
                        flow.dst_port,
                    )
                )
                self.try_associate(dst_endpoint, flow.sock_id, ValueType.SOCK_ID)
                return True
                # return False

    def add_flow_with_dst_endpoint(self, dst_ip, dst_port, flow_id):
        dst_endpoint = (dst_ip, dst_port)
        # print("add_flow_with_dst_endpoint:", dst_endpoint, flow_id)
        if not dst_endpoint in self.dst_endpoint_to_flow_id:
            self.dst_endpoint_to_flow_id[dst_endpoint] = flow_id
            # print("endpoint to flow_id:", self.dst_endpoint_to_flow_id)
            log.debug(
                "register env {} flow: {} with dst_endpoint: {}:{}".format(
                    self.env_id, flow_id, ipaddress.IPv4Address(dst_ip), dst_port
                )   
            )
            self.try_associate(dst_endpoint, flow_id, ValueType.FLOW_ID)
            
    def add_flowid_with_sockid(self, flow_id, sock_id):
        if not flow_id in self.flow_id_to_sock_id:
            self.flow_id_to_sock_id[flow_id] = sock_id
            log.debug(
                "register env {} flow: {} with sock_id: {}".format(
                    self.env_id, flow_id, sock_id
                )
            )
            
    def get_flowId_by_sockId(self, sock_id):
        if sock_id in self.sock_id_to_flow_id.keys():
            return self.sock_id_to_flow_id[sock_id]
        else:
            return None

    def get_sockId_by_flowId(self, flow_id):
        if flow_id in self.flow_id_to_sock_id.keys():
            return self.flow_id_to_sock_id[flow_id]
        else:
            return None

    def get_flowId_by_endpoint(self, dst_ip, dst_port):
        dst_endpoint = (dst_ip, dst_port)
        if dst_endpoint in self.dst_endpoint_to_flow_id.keys():
            return self.dst_endpoint_to_flow_id[dst_endpoint]
        else:
            return None

    def remove_flow_by_sockId(self, sock_id) -> Optional[tuple]:
        if sock_id in self.kernel_flows:
            # they should exists in these two maps
            flow = self.kernel_flows[sock_id]
            dst_endpoint = (flow.dst_ip, flow.dst_port)
            if dst_endpoint in self.dst_endpoint_to_flow_id:
                self.dst_endpoint_to_flow_id.pop(dst_endpoint)
            if dst_endpoint in self.dst_endpoint_to_sock_id:
                self.dst_endpoint_to_sock_id.pop(dst_endpoint)
            if sock_id in self.sock_id_to_flow_id.keys():
                self.sock_id_to_flow_id.pop(sock_id)
            log.debug("remove kernel flow: {} for env {}".format(sock_id, self.env_id))
            self.kernel_flows.pop(sock_id)
            return dst_endpoint
        # we might receive the lazy kernel release message from kernel
        return None

    def remove_flow_by_flowId(self, flow_id) -> Optional[tuple]:
        if flow_id in self.flow_id_to_sock_id:
            log.debug("remove env: {} flow: {}".format(self.env_id, flow_id))
            sock_id = self.flow_id_to_sock_id.pop(flow_id)
            # as the kernel flow release use the lazy one, here we also remove the kernel flow info
            return self.remove_flow_by_sockId(sock_id)
        return None

    def remove_all_env_flows(self):
        for flow_id in self.flow_id_to_sock_id.copy():
            self.flow_id_to_sock_id.pop(flow_id)


class EnvFlows(object):
    def __init__(self):
        self.env_id = None
        # hash of env id
        self.h_id = None
        self.flows_per_env = dict()

        # cache a map from (dst_ip, dst_port) tuple to env_id 
        self.dst_endpoint_to_env_id = dict()
        self.sock_id_to_env_id = dict()

    def register_env(self, env_id):
        self.env_id = env_id
        self.h_id = hash(self.env_id)
        self.flows_per_env[self.h_id] = ActiveFlowMap()
        # for logging convinience
        self.flows_per_env[self.h_id].env_id = env_id

    def get_env_flows(self, env_id) -> Optional[ActiveFlowMap]:
        id = hash(env_id)
        if id in self.flows_per_env:
            return self.flows_per_env[id]
        else:
            return None

    def release_env(self, env_id):
        log.info("env {} notifies spine to terminate, remove all flows".format(env_id))
        id = hash(env_id)

        if id in self.flows_per_env:
            self.flows_per_env.pop(id)

        # remove cache items
        self.dst_endpoint_to_env_id = {key:val for key, val in self.dst_endpoint_to_env_id.items() if val != env_id}
        self.sock_id_to_env_id = {key:val for key, val in self.sock_id_to_env_id.items() if val != env_id}

    def bind_endpoint_to_env(self, dst_ip, dst_port, env_id):
        dst_endpoint = (dst_ip, dst_port)
        self.dst_endpoint_to_env_id[dst_endpoint] = env_id
        log.info("bind endpoint: {}:{} with env: {}".format(ipaddress.IPv4Address(dst_ip), dst_port, env_id))

    def bind_sock_id_to_env(self, sock_id, env_id):
        # if sock_id not in self.sock_id_to_env_id:
        self.sock_id_to_env_id[sock_id] = env_id
        log.info("bind kernel flow: {} with env: {}".format(sock_id, env_id))
    
    def release_endpoint_to_env(self, dst_ip, dst_port):
        dst_endpoint = (dst_ip, dst_port)
        if dst_endpoint in self.dst_endpoint_to_env_id:
            self.dst_endpoint_to_env_id.pop(dst_endpoint)

    def release_sock_id_to_env(self, sock_id):
        if sock_id in self.sock_id_to_env_id:
            self.sock_id_to_env_id.pop(sock_id)
