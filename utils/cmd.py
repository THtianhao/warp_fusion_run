import subprocess

def gitclone(url, recursive=False, dest=None):
    command = ['git', 'clone', url]
    if dest: command.append(dest)
    if recursive: command.append('--recursive')
    res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)


def pipi(modulestr):
    res = subprocess.run(['python','-m','pip', '-q', 'install', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)


def pipie(modulestr):
    res = subprocess.run(['python','-m','pip', 'install', '-e', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def pipis(modulestrs:list):
    pip_cmd = ['python','-m','pip', '-q', '-y','install']
    pips_cmd = pip_cmd + modulestrs
    res = subprocess.run(pips_cmd, stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def gite(modulestr):
    res = subprocess.run(['git', 'install', '-e', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def wget_p(url, outputdir):
    res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)


def get_version(package):
    proc = subprocess.run(['pip','show', package], stdout=subprocess.PIPE)
    out = proc.stdout.decode('UTF-8')
    returncode = proc.returncode
    if returncode != 0:
        return -1
    return out.split('Version:')[-1].split('\n')[0]

