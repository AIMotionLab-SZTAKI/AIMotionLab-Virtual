## Connect via Multicast. This version works.

import socket as socket
import struct as struct

SOCKET_BUFSIZE = 0x100000

client_address = '192.168.1.66'
server_address = '192.168.1.141'  ## Motive Tracker set to Multicast
multicast_address = '239.255.42.99'
command_port = 1510
data_port = 1511

# Bind client address at data port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
sock.bind((client_address, data_port))
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFSIZE)

# Add the client IP address to the multicast group
mreq = struct.pack("=4s4s",
                   socket.inet_aton(multicast_address),
                   socket.inet_aton(client_address))
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

# Set to non-blocking
sock.setblocking(0)

while True:
    try:
        msg, address = sock.recvfrom(SOCKET_BUFSIZE)
    except:
        pass
    else:
        print(msg, "\n")