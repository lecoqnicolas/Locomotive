/opt/tritonserver/bin/tritonserver  --model-repository=/models --grpc-use-ssl=1 --grpc-use-ssl-mutual=1 --grpc-server-cert /opt/tritonserver/certs/server_localhost.crt --grpc-server-key /opt/tritonserver/certs/server_localhost.key --grpc-root-cert /opt/tritonserver/certs/ca_localhost.crt --allow-http False --log-file /opt/tritonserver/logs.txt