#! /usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR="${SCRIPT_DIR/\/userspace\//\/dataspace\/}"
DRIVE_DIR="${DATA_DIR#/workspace}"

DOWNLOAD_EXE="/workspace/userspace/_shared/download.sh"
source /workspace/userspace/_shared/colors.sh


# directories creation
mkdir -p "${DATA_DIR}"


SECTION "Updating package lists..."
sudo apt update
sudo apt install -y g++

SECTION "Installing python libraries..."
pip install -r ${SCRIPT_DIR}/requirements.txt


echo -e "${GREEN}Done.${NC}"