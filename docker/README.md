# pKasso Web Server Docker Image

Build the image from the repository root:

```bash
docker/build.sh
```

Run the web server:

```bash
docker/run.sh
```

Then open <http://localhost:8001>.

The container starts:

```bash
pkasso-web --host 0.0.0.0 --port 8001
```

To use a different host port:

```bash
PORT=8080 docker/run.sh
```

To serve pKasso behind a reverse proxy at a path prefix such as
`https://example.com/pkasso/`, pass the public prefix to the container:

```bash
PKASSO_ROOT_PATH=/pkasso docker/run.sh
```

The reverse proxy should strip `/pkasso/` before forwarding requests to the
container. If the proxy terminates HTTPS and forwards to the container over
HTTP, it should also pass the original protocol and host:

```nginx
proxy_set_header Host $host;
proxy_set_header X-Forwarded-Host $host;
proxy_set_header X-Forwarded-Proto https;
proxy_set_header X-Forwarded-Port 443;
```

Allow pKasso to trust those forwarded headers from the proxy. For a container
that is reachable only through the trusted reverse proxy, this can be:

```bash
PKASSO_ROOT_PATH=/pkasso PKASSO_FORWARDED_ALLOW_IPS='*' docker/run.sh
```

For stricter deployments, set `PKASSO_FORWARDED_ALLOW_IPS` to the proxy IP or a
comma-separated list of proxy IPs.

To transfer the image to another machine:

```bash
docker/save.sh
```

Copy `pkasso-webserver.tar` to the other machine, then run:

```bash
docker/load.sh pkasso-webserver.tar
docker/run.sh
```
