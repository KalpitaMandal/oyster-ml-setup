FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get install apt-utils -y
RUN apt-get install python3 python3-pip python3-dev net-tools iptables iproute2 curl wget -y

RUN pip install flask

WORKDIR /app

# supervisord to manage programs
RUN wget -O supervisord http://public.artifacts.marlin.pro/projects/enclaves/supervisord_master_linux_amd64
RUN chmod +x supervisord

# transparent proxy component inside the enclave to enable outgoing connections
RUN wget -O ip-to-vsock-transparent http://public.artifacts.marlin.pro/projects/enclaves/ip-to-vsock-transparent_v1.0.0_linux_amd64
RUN chmod +x ip-to-vsock-transparent

# key generator to generate static keys
RUN wget -O keygen http://public.artifacts.marlin.pro/projects/enclaves/keygen_v1.0.0_linux_amd64
RUN chmod +x keygen

# attestation server inside the enclave that generates attestations
RUN wget -O attestation-server http://public.artifacts.marlin.pro/projects/enclaves/attestation-server_v1.0.0_linux_amd64
RUN chmod +x attestation-server

# proxy to expose attestation server outside the enclave
RUN wget -O vsock-to-ip http://public.artifacts.marlin.pro/projects/enclaves/vsock-to-ip_v1.0.0_linux_amd64
RUN chmod +x vsock-to-ip

# dnsproxy to provide DNS services inside the enclave
RUN wget -O dnsproxy http://public.artifacts.marlin.pro/projects/enclaves/dnsproxy_v0.46.5_linux_amd64
RUN chmod +x dnsproxy

RUN wget -O oyster-keygen http://public.artifacts.marlin.pro/projects/enclaves/keygen-secp256k1_v1.0.0_linux_amd64
RUN chmod +x oyster-keygen

# supervisord config
COPY supervisord.conf /etc/supervisord.conf

COPY linear_reg.py ./

COPY mnist_97.pth ./

COPY input.json ./

COPY model.sav ./

# setup.sh script that will act as entrypoint
COPY setup.sh ./
RUN chmod +x setup.sh

COPY http_proxy ./
RUN chmod +x http_proxy

# entry point
ENTRYPOINT [ "/app/setup.sh" ]
