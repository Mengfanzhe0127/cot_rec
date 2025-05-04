#!/bin/bash

REPO_URL="https://github.com/Mengfanzhe0127/cot_rec.git"
RETRY_INTERVAL=300
MAX_RETRIES=100

retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    git push
    if [ $? -eq 0 ]; then
        echo "Push successful!"
        exit 0
    else
        error_message=$(git push 2>&1)
        if [[ $error_message == *"GnuTLS recv error (-110)"* ]]; then
            echo "Fatal error encountered: GnuTLS recv error (-110)"
            echo "Retrying in $RETRY_INTERVAL seconds..."
            sleep $RETRY_INTERVAL
            retry_count=$((retry_count + 1))
        else
            echo "Push failed with a different error:"
            echo "$error_message"
            sleep $RETRY_INTERVAL
            retry_count=$((retry_count + 1))
        fi
    fi
done

echo "Max retries reached. Push failed."
exit 1