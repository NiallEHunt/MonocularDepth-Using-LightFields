from azureml.core import Workspace

# Removed details for the workspace. You will have to set this up if you want to use AzureML
ws = Workspace.create(name='',
                      subscription_id='',
                      resource_group='',
                      create_resource_group=False,
                      location='')

ws.write_config(path='.azureml')