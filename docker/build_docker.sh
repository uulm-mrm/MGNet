#!/bin/bash
# read arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -d|--dev)
    BUILD_DEV_CONTAINER=true
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
    *)  # unknown option
    echo "Unknown option passed to build_docker.sh"
    SHOW_HELP=true
    break
    ;;
  esac
done

if [ -z ${TAG} ]; then
  echo "No tag given. Defaulting to latest"
  TAG="latest"
fi

# show help
if [ "$SHOW_HELP" = true ]; then
  echo "Usage: ./build_docker.sh [Options]"
  echo ""
  echo "Options:"
  echo " * -d|--dev:       Build development container with user based library installation"
  echo " * -t|--tag TAG:   Image tag (default: latest)"
  echo " * -h|--help:      Show this message"
  echo ""
  exit 1
fi

if [ "$BUILD_DEV_CONTAINER" = true ]; then
  if [[ "$(docker images -q mgnet:${TAG} 2> /dev/null)" == "" ]]; then
    echo "Base image missing. Build base image first..."
    docker build -t mgnet:${TAG} .
  fi
  docker build --build-arg TAG=${TAG} --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t mgnet-dev:${TAG} -f develop.Dockerfile .
else
  docker build -t mgnet:${TAG} .
fi