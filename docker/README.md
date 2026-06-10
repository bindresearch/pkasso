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

To transfer the image to another machine:

```bash
docker/save.sh
```

Copy `pkasso-webserver.tar` to the other machine, then run:

```bash
docker/load.sh pkasso-webserver.tar
docker/run.sh
```
