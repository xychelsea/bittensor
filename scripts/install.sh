#!/bin/bash
set -u

abort() {
  printf "%s\n" "$1"
  exit 1
}

getc() {
  local save_state
  save_state=$(/bin/stty -g)
  /bin/stty raw -echo
  IFS= read -r -n 1 -d '' "$@"
  /bin/stty "$save_state"
}

wait_for_user() {
  local c
  echo
  echo "Press RETURN to continue or any other key to abort"
  getc c
  # we test for \r and \n because some stuff does \r instead
  if ! [[ "$c" == $'\r' || "$c" == $'\n' ]]; then
    exit 1
  fi
}

shell_join() {
  local arg
  printf "%s" "$1"
  shift
  for arg in "$@"; do
    printf " "
    printf "%s" "${arg// /\ }"
  done
}

# string formatters
if [[ -t 1 ]]; then
  tty_escape() { printf "\033[%sm" "$1"; }
else
  tty_escape() { :; }
fi
tty_mkbold() { tty_escape "1;$1"; }
tty_underline="$(tty_escape "4;39")"
tty_blue="$(tty_mkbold 34)"
tty_red="$(tty_mkbold 31)"
tty_bold="$(tty_mkbold 39)"
tty_reset="$(tty_escape 0)"

ohai() {
  printf "${tty_blue}==>${tty_bold} %s${tty_reset}\n" "$(shell_join "$@")"
}

# Things can fail later if `pwd` doesn't exist.
# Also sudo prints a warning message for no good reason
cd "/usr" || exit 1

linux_install_pre() {
    sudo apt-get update 
    sudo apt-get install --no-install-recommends --no-install-suggests -y apt-utils curl git cmake build-essential 
}

linux_install_python() {
    which -s python3.8
    if [[ $? != 0 ]] ; then
        ohai "Installing python3.8"
        sudo apt-get install --no-install-recommends --no-install-suggests -y python3.8
    else
        ohai "Updating python3.8"
        sudo apt-get update python3.8
    fi
    ohai "Installing python tools"
    sudo apt-get install --no-install-recommends --no-install-suggests -y python3-pip python3.8-dev python3.8-venv
}

linux_activate_installed_python() {
    ohai "Creating python virtualenv"
    mkdir -p ~/.bittensor/bittensor
    cd ~/.bittensor/
    python3.8 -m venv env
    ohai "Entering bittensor-environment"
    source env/bin/activate
    ohai "You are using python@3.8$"
    ohai "Installing python tools"
    python -m pip install --upgrade pip
    python -m pip install python-dev
}