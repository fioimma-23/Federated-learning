from scapy.all import rdpcap

# Read
packets = rdpcap('newpcap.pcap')

flows = {}
for pkt in packets:
    if pkt.haslayer('IP'):
        src_ip = pkt['IP'].src
        dst_ip = pkt['IP'].dst
        proto = pkt['IP'].proto

        #TCP or UDP layer
        if pkt.haslayer('TCP') or pkt.haslayer('UDP'):
            if pkt.haslayer('TCP'):
                src_port = pkt['TCP'].sport
                dst_port = pkt['TCP'].dport
            else:
                src_port = pkt['UDP'].sport
                dst_port = pkt['UDP'].dport

            #Create flow ID
            flow_id = (src_ip, dst_ip, src_port, dst_port, proto)

            #Add packet to flow
            if flow_id not in flows:
                flows[flow_id] = []
            flows[flow_id].append(pkt)

#Extract headers from packets
def extract_headers(flow_packets, header_size=54):
    headers = []
    for pkt in flow_packets:
        raw_bytes = bytes(pkt)[:header_size]
        headers.append(raw_bytes)
    return headers  

#Extract headers
flow_headers = {}
for flow_id, flow_pkts in flows.items():
    headers = extract_headers(flow_pkts)
    flow_headers[flow_id] = headers

#Anonymize headers
def anonymize_header(header_bytes):
    header = bytearray(header_bytes)
    header[26:34] = b'\x00' * 8   # IP addresses
    header[0:12] = b'\x00' * 12   # MAC addresses
    header[34:38] = b'\x00' * 4    # Sequence numbers
    header[24:26] = b'\x00' * 2    # Identification field
    return bytes(header)

for flow_id in flow_headers:
    headers = flow_headers[flow_id]
    anonymized_headers = [anonymize_header(hdr) for hdr in headers]
    flow_headers[flow_id] = anonymized_headers

#Convert header to numeric values
def header_to_numeric(header_bytes):
    return [int(b) for b in header_bytes]

#Create numeric headers
flow_features = {}
for flow_id, headers in flow_headers.items():
    numeric_headers = [header_to_numeric(hdr) for hdr in headers]
    flow_features[flow_id] = numeric_headers

#Flow analysis
def analyze_flows(flows):
    analysis_results = {}
    for flow_id, flow_pkts in flows.items():
        packet_count = len(flow_pkts)
        total_size = sum(len(pkt) for pkt in flow_pkts)
        avg_packet_size = total_size / packet_count if packet_count > 0 else 0
        
        analysis_results[flow_id] = {
            'packet_count': packet_count,
            'total_size': total_size,
            'avg_packet_size': avg_packet_size,
        }
    return analysis_results

flow_analysis = analyze_flows(flows)

#results
print(f"Total number of unique flows: {len(flows)}")
for flow_id, analysis in flow_analysis.items():
    print(f"Flow ID: {flow_id}, "
          f"Packet Count: {analysis['packet_count']}, "
          f"Total Size: {analysis['total_size']} bytes, "
          f"Average Packet Size: {analysis['avg_packet_size']} bytes")
