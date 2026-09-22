import rclpy, sys, time
from rclpy.node import Node
from aimdk_msgs.srv import MigrateSystemState, GetSystemState
target=sys.argv[1]
rclpy.init(); n=Node("ffmaster_migrate")
gc=n.create_client(GetSystemState,"/aimdk_5Fmsgs/srv/GetSystemState")
mc=n.create_client(MigrateSystemState,"/aimdk_5Fmsgs/srv/MigrateSystemState")
def cur(timeout=8.0):
    if not gc.wait_for_service(timeout_sec=timeout): return "<service unavailable>"
    f=gc.call_async(GetSystemState.Request())
    rclpy.spin_until_future_complete(n,f,timeout_sec=timeout)
    return f.result().cur_state if f.result() else "<no response>"
print("  current state : %s" % cur(), flush=True)
if not mc.wait_for_service(timeout_sec=8.0):
    print("  MigrateSystemState UNAVAILABLE"); sys.exit(1)
req=MigrateSystemState.Request(); req.state=target
print("  requesting    : %s" % target, flush=True)
t0=time.time()
f=mc.call_async(req)
rclpy.spin_until_future_complete(n,f,timeout_sec=30.0)
r=f.result()
if r is None: print("  *** NO RESPONSE (timeout) ***")
else:
    print("  response code : %s" % r.header.header.code)
    print("  message       : %s" % r.header.message)
print("  elapsed       : %.1f s" % (time.time()-t0), flush=True)
for i in range(10):
    time.sleep(1.0); s=cur(4.0)
    print("  t+%2ds state   : %s" % (i+1, s), flush=True)
    if s==target: break
n.destroy_node(); rclpy.shutdown()
