# certification authority
openssl genrsa -passout pass:1234 -des3 -out certs/ca_localhost.key 4096
openssl req -passin pass:1234 -new -x509 -days 365 -key certs/ca_localhost.key -out certs/ca_localhost.crt -subj  "/C=FR/ST=France/L=Gra/O=MEAETRAD/OU=MEAETRAD/CN=Root CA"
# server key and certs
openssl genrsa -passout pass:1234 -des3 -out certs/server_localhost.key 4096
openssl req -passin pass:1234 -new -key certs/server_localhost.key -out certs/server_localhost.csr -subj  "/C=FR/ST=France/L=Gra/O=MEAETRAD/OU=MEAETRAD/CN=localhost"
openssl x509 -req -passin pass:1234 -days 365 -in certs/server_localhost.csr -CA certs/ca_localhost.crt -CAkey certs/ca_localhost.key -set_serial 01 -out certs/server_localhost.crt
openssl rsa -passin pass:1234 -in certs/server_localhost.key -out certs/server_localhost.key
# client keey and certs
openssl genrsa -passout pass:1234 -des3 -out certs/client_localhost.key 4096
openssl req -passin pass:1234 -new -key certs/client_localhost.key -out certs/client_localhost.csr -subj  "/C=FR/ST=France/L=Gra/O=MEAETRAD/OU=MEAETRAD/CN=localhost"
openssl x509 -passin pass:1234 -req -days 365 -in certs/client_localhost.csr -CA certs/ca_localhost.crt -CAkey certs/ca_localhost.key -set_serial 01 -out certs/client_localhost.crt
openssl rsa -passin pass:1234 -in certs/client_localhost.key -out certs/client_localhost.key
