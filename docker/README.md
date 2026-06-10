# pKasso Webserver Docker Image

Build the image from the repository root:

```bash
docker/build.sh
```

The build uses Docker BuildKit SSH forwarding because `pyproject.toml` includes private GitHub dependencies such as `git+ssh://git@github.com/...`.
Make sure your SSH agent can access those repositories before building:

```bash
ssh-add -l
ssh -T git@github.com
```

If these checks pass but the Docker build still reports `Permission denied (publickey)`, make sure you are not running the build with `sudo`, and confirm your shell has an SSH agent socket:

```bash
echo "$SSH_AUTH_SOCK"
```

You can also test the exact URL style used by `pip`:

```bash
git ls-remote ssh://git@github.com/bindresearch/pkasso.git
```

Run the webserver:

```bash
docker/run.sh
```

Then open <http://localhost:8001>.

To pass additional Docker runtime flags, set `GPU_ARGS`:

```bash
GPU_ARGS="--gpus all" docker/run.sh
```

To transfer the image to another computer:

```bash
docker/save.sh
```

Copy `pkasso-webserver.tar` to the other computer, then run:

```bash
docker/load.sh pkasso-webserver.tar
docker/run.sh
```

The container starts the app with:

```bash
pkasso-web --host 0.0.0.0 --port 8001
```
