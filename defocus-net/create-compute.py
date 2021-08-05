from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

ws = Workspace.from_config()

cluster_name = "gpu-cluster"

# Verify that the cluster does not exist already
try:
    gpu_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                           idle_seconds_before_scaledown=2400,
                                                           min_nodes=0,
                                                           max_nodes=4)
    gpu_cluster = ComputeTarget.create(ws, cluster_name, compute_config)

gpu_cluster.wait_for_completion(show_output=True)

print(gpu_cluster.get_status().serialize())
