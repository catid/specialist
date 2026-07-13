#!/bin/bash
# One-shot RAM fault diagnostic. Run after each swap/reboot; compare output.
echo "===== $(date) ====="
echo "--- usable memory (7x128GiB = ~880GiB, 8x = ~1006GiB) ---"
awk '/MemTotal/{printf "MemTotal: %.1f GiB\n",$2/1048576}' /proc/meminfo
echo
echo "--- DIMM serial present in each channel (A-H); missing = mapped out ---"
sudo -n dmidecode -t memory 2>/dev/null | awk '
/Memory Device/{d=1;loc="";size="";ser=""}
d&&/Bank Locator:/{sub(/.*CHANNEL /,"");ch=$1}
d&&/^\tSize:/{sub(/^\tSize: /,"");size=$0}
d&&/Serial Number:/{sub(/.*Serial Number: /,"");ser=$0;
  printf "  CHANNEL %s : %-22s serial=%s\n",ch,size,ser; d=0}'
echo
echo "--- firmware fatal-error record (which channel got mapped out) ---"
sudo -n dmesg 2>/dev/null | grep -E "Locator: P0M|error_status|physical memory map-out|event severity" | sed 's/\[[0-9. ]*\]//'
[ -z "$(sudo -n dmesg 2>/dev/null | grep 'map-out')" ] && echo "  (no map-out event this boot)"
echo
echo "--- EDAC live ECC error counts (should stay 0) ---"
for f in /sys/devices/system/edac/mc/mc*/ce_count /sys/devices/system/edac/mc/mc*/ue_count; do
  [ -f "$f" ] && echo "  $f = $(cat $f)"; done
