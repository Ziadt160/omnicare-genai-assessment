#!/bin/sh
# Inject the API base URL at start rather than baking it in at build time -
# otherwise the image is tied to one hostname and cannot be reused.
set -e
: "${API_BASE:=http://localhost:8080}"
cat > /usr/share/nginx/html/config.js <<CONFIG
window.OMNICARE_API = "${API_BASE}";
CONFIG
exec nginx -g 'daemon off;'
