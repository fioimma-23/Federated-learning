import pyshark

#Load
capture = pyshark.FileCapture('newpcap.pcap')

#Iterate
for packet in capture:
    try:
        src = packet.ip.src
        dst = packet.ip.dst
        protocol = packet.transport_layer
        length = packet.length
        
        print(f"Source: {src}, Destination: {dst}, Protocol: {protocol}, Length: {length}")
    except AttributeError:
        print("Packet has no IP layer.")

capture.close()
