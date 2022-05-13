#!/usr/bin/env bash
# read arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  if [ -z ${EXEC_ARGS+x} ]; then
    case $key in
      -d|--dev)
      RUN_DEV_CONTAINER=true
      shift # past argument
      ;;
      -b|--bind_zsh)
      BIND_ZSH=true
      shift # past argument
      ;;
      -s|--bind_sources)
      BIND_SOURCES=true
      shift # past argument
      ;;
      -t|--tag)
      TAG="$2"
      shift # past argument
      shift # past value
      ;;
      -h|--help)
      SHOW_HELP=true
      break
      ;;
      --exec)
      EXEC_ARGS=("")
      shift # past argument
      ;;
      *)  # unknown options are passed to docker
      CMD_EXTRA_ARGS+=("$1")
      shift # past argument
      ;;
    esac
  else
    EXEC_ARGS+=("$1") # save it in an array for later
    shift
  fi
done

# check args
if [[ ("$BIND_ZSH" = true || "$BIND_SOURCES" = true) && "$RUN_DEV_CONTAINER" != true ]]; then
  echo "Invalid combination of options!"
  SHOW_HELP=true
fi

# show help
if [ "$SHOW_HELP" = true ]; then
  echo "Usage: ./run_docker.sh [Options] [Arguments]"
  echo ""
  echo "Options:"
  echo " * -d|--dev:             Run development container"
  echo " * -b|--bind_zsh:        Bind user zsh setup into docker container (Only supported in dev container)"
  echo " * -s|--bind_sources:    Bind local MGNet sources into docker container (Only supported in dev container)"
  echo " * -t|--tag TAG:         Image tag (default: latest version)"
  echo " * --exec ARG [ARG ..]:  All subsequent arguments are executed within the Docker container"
  echo " * -h|--help:            Show this message"
  echo ""
  exit 1
fi

if [ -z ${TAG} ]; then
  echo "No tag given. Defaulting to latest"
  TAG="latest"
fi

# use shell in case no exec args are given
if [ ${#EXEC_ARGS[@]} -eq 0 ]; then
  EXEC_ARGS=(${SHELL})
fi

if [ "$RUN_DEV_CONTAINER" = true ]; then
  IMAGE=mgnet-dev
else
  IMAGE=mgnet
fi

DOCKER_ARGS=(
  -v /tmp/.X11-unix:/tmp/.X11-unix
  -e DOCKER_MACHINE_NAME="${IMAGE}:${TAG}"
  --network=host
  --ulimit core=99999999999:99999999999
  --ulimit nofile=1024
  --privileged
  --rm
  -e DISPLAY=$DISPLAY
  -e QT_X11_NO_MITSHM=1
  --ipc=host
  ${CMD_EXTRA_ARGS[@]}
)

if [ "$BIND_ZSH" = true ]; then
  DOCKER_ARGS+=(
    -v $HOME/.aliases:/home/appuser/.aliases
    -v $HOME/.fzf.zsh:/home/appuser/.fzf.zsh
    -v $HOME/.fzf:/home/appuser/.fzf
    -v $HOME/.fzf:$HOME/.fzf  # fzf uses hardcoded paths
    -v $HOME/.oh-my-zsh:/home/appuser/.oh-my-zsh
    -v $HOME/.oh-my-zsh:$HOME/.oh-my-zsh  # oh-my-zsh sometimes uses hardcoded paths
    -v $HOME/.p10k.zsh:/home/appuser/.p10k.zsh
    -v $HOME/.zsh_history:/home/appuser/.zsh_history
    -v $HOME/.zshrc:/home/appuser/.zshrc
  )
fi

if [ "$BIND_SOURCES" = true ]; then
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
  MGNET_DIR="$(readlink -f "${SCRIPT_DIR}/../")"
  DOCKER_ARGS+=(
    -v $MGNET_DIR:/home/appuser/MGNet
  )
fi

docker run --gpus all ${DOCKER_ARGS[@]} -it ${IMAGE}:${TAG} ${EXEC_ARGS[@]}
