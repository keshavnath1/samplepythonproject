import subprocess


# Function to run a command and print its output
def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()

    # Print standard output and error
    if stdout:
        print(stdout.decode())
    if stderr:
        print(stderr.decode())


# Enable verbose mode by setting the shell option
verbose_shell = True

# Run pytest with detailed output
run_command('pytest test_temporal_model.py -s')


# Run pytest to generate JSON report
run_command('pytest --json-report --json-report-file=cucumber_report.json')

# Run the Python script to generate the HTML report
run_command('python ./reports/prepare_html_cucumber.py')

print("Automation script executed successfully.")
