import os
import sys

def redirect_output_to_new_run(run_directory='.', prefix='run'):
    files = os.listdir(run_directory)
    max_i = -1
    for f in files:
        if f.startswith('run'):
            max_i = max(max_i, int(f.split('_')[1].split('.')[0]))
    
    next_i = max_i+1
    run_path = os.path.join(run_directory, 'run_%04i.out'%next_i)
    print('Redirecting Output To: %s'%run_path)
    file_handle = open(run_path, 'w')
    sys.stdout = file_handle
    sys.stderr = file_handle
