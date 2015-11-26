import socket
import os
import sys

HOST = "127.0.0.1"
PORT = 2000
BYTES_SIZE = 1024
OPTIONS = {1, 2, 3}
MESSAGE = "Hello, server";

def options():
    print "1 - Send Message \n"
    print "2 - Read messages that was set \n"
    print "3 - Close client"

def doAction(_type, sk):
    if _type is 1:
        MESSAGE = raw_input("Send message: ")
        sk.sendall(MESSAGE)
    elif _type is 2:
        data = sk.recv(BYTES_SIZE)
        print 'Server Message: ', repr(data) 
    else:
        print 'Close server...' 
        sk.close()
        sys.exit()
        
        

if __name__ == '__main__':
    opt = -1;
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.connect((HOST, PORT))
    sk.sendall(MESSAGE)
    data = sk.recv(BYTES_SIZE)
    print 'Server Message: ', repr(data) 
    while opt != 0:
        opt = int(raw_input('Entre com uma opcao : '))
        if opt in OPTIONS:
            doAction(opt, sk)
        else:
            options()
    sys.exit()