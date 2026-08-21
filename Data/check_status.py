import ee

# Initialize Earth Engine
ee.Initialize(project='spring-radar-478010-k1')

# Fetch the 3 most recent tasks
for task in ee.batch.Task.list()[:3]:
    name = task.config.get('description', 'Unnamed Task')
    print(f"Task: {name} | Status: {task.state}")
    if task.state == 'FAILED':
        print(f"Error: {task.status().get('error_message')}")
        print("-" * 40)