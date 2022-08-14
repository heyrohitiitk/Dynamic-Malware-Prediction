import logging
import json
import psutil

from lib.common.abstracts import Auxiliary
from lib.common.results import NetlogFile

log = logging.getLogger(__name__)

class Custom(Auxiliary):
    """Gather custom data"""

    def __init__(self, options={}, analyzer=None):
        Auxiliary.__init__(self, options, analyzer)

    def start(self):
	log.info("Starting my Custom auxiliary module")
	log.info("hit custom.py start")
	pids = psutil.pids()
	max_pid = max(pids)
	total_id = len(pids)

	psutil.cpu_times_percent()
	cputimes = psutil.cpu_times_percent(0.1)
	usertime = cputimes.user
	systemtime = cputimes.system

	memuse = psutil.virtual_memory().used
	swapuse = psutil.swap_memory().used

	net = psutil.net_io_counters()

	packsent = net.packets_sent
	packrecv = net.packets_recv

	bytessent = net.bytes_sent
	bytesrecv = net.bytes_recv
	result=[max_pid,total_id,usertime,systemtime,memuse,swapuse,packsent,packrecv,bytessent,bytesrecv]
        
	nf = NetlogFile("logs/initial.json")
	
	log.info("result_log_info_hit  "+str(result))
	nf.send(json.dumps(['foo', {'bar':result}]))

