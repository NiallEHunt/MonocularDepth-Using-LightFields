from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets/defocus-net'))

    experiment = Experiment(workspace=ws, name='defocus-net-arch1-epoch101')

    config = ScriptRunConfig(
        source_directory='./src',
        script='train.py',
        compute_target='gpu-cluster',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount(),
            '--epochs', 1000],
    )
    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='pytorch-env',
        file_path='./.azureml/pytorch-env.yml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)