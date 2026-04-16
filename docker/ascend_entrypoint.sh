#!/usr/bin/env bash
set -e

append_path_if_dir() {
  local var_name="$1"
  local dir_path="$2"
  if [ -d "${dir_path}" ]; then
    # shellcheck disable=SC2086
    eval "current_value=\${${var_name}:-}"
    case ":${current_value}:" in
      *":${dir_path}:"*) ;;
      *)
        if [ -n "${current_value}" ]; then
          eval "export ${var_name}=\"${dir_path}:${current_value}\""
        else
          eval "export ${var_name}=\"${dir_path}\""
        fi
        ;;
    esac
  fi
}

# Load Ascend runtime env if available.
if [ -f "/etc/profile.d/ascend-env.sh" ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/ascend-env.sh
elif [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f "/root/Ascend/ascend-toolkit/set_env.sh" ]; then
  # shellcheck disable=SC1091
  source /root/Ascend/ascend-toolkit/set_env.sh
fi

# Add common Ascend driver tool paths for npu-smi.
append_path_if_dir PATH "/usr/local/Ascend/driver/tools"
append_path_if_dir PATH "/usr/local/Ascend/driver/bin"
append_path_if_dir PATH "/usr/local/sbin"
append_path_if_dir PATH "/usr/sbin"

# Add common Ascend driver/runtime library paths.
append_path_if_dir LD_LIBRARY_PATH "/usr/local/Ascend/driver/lib64"
append_path_if_dir LD_LIBRARY_PATH "/usr/local/Ascend/driver/lib64/common"
append_path_if_dir LD_LIBRARY_PATH "/usr/local/Ascend/driver/lib64/driver"
append_path_if_dir LD_LIBRARY_PATH "/usr/local/Ascend/driver/lib64/stub"
append_path_if_dir LD_LIBRARY_PATH "/usr/local/dcmi/lib64"
append_path_if_dir LD_LIBRARY_PATH "/usr/lib64"

# Helpful warning for debug, but do not block container startup.
if ! command -v npu-smi >/dev/null 2>&1; then
  echo "[ascend-entrypoint] WARNING: npu-smi not found in PATH."
fi

if ! ldconfig -p 2>/dev/null | grep -q "libascend_hal.so"; then
  echo "[ascend-entrypoint] WARNING: libascend_hal.so not found by dynamic linker cache."
fi

exec "$@"
