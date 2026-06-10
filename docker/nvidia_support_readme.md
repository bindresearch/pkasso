# Docker GPU Support Setup (NVIDIA Container Toolkit)

This guide explains how to install and configure Docker with GPU support using the NVIDIA Container Toolkit on Ubuntu.

---

## Prerequisites

* Ubuntu system
* Docker installed
* NVIDIA GPU with drivers installed

Verify GPU driver:

```bash
nvidia-smi
```

If this fails, fix your NVIDIA driver installation before continuing.

---

## 1. Add NVIDIA Container Toolkit Repository

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

---

## 2. Install NVIDIA Container Toolkit

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

---

## 3. Configure Docker Runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 4. Verify Installation

Check toolkit binary:

```bash
which nvidia-ctk
```

Check Docker runtimes:

```bash
docker info | grep -i runtime
```

Expected output should include `nvidia`.

---

## 5. Test GPU in Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed inside the container.

---

## Troubleshooting

### Package not found (`nvidia-container-toolkit`)

* Ensure the NVIDIA repository was added correctly
* Run `sudo apt-get update` again

### `nvidia-ctk` not found

* Toolkit installation likely failed

### Docker shows only `runc`

* Re-run:

  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

### GPU not available in container

* Verify host GPU:

  ```bash
  nvidia-smi
  ```
* Ensure correct driver is installed

---

## Reference

* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

---

