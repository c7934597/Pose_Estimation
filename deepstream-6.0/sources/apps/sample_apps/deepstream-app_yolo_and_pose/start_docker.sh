#!/bin/bash
xhost + && docker exec -it -d 628f50f8de63 bash /etc/init.d/test.sh

exit 0
