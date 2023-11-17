import json
from jinja2 import Environment, FileSystemLoader

# Load JSON data
with open('cucumber_report.json', 'r') as file:
    report_data = json.load(file)

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('reports/report_template.html')

# Render the template with data
html_output = template.render(report=report_data)

# Write the output to a file
with open('cucumber_report.html', 'w') as file:
    file.write(html_output)

print("Report generated: cucumber_report.html")
