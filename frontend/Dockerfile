FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY src/ /usr/share/nginx/html/
COPY docker-entrypoint.sh /docker-entrypoint-omnicare.sh

RUN chmod +x /docker-entrypoint-omnicare.sh

EXPOSE 80
ENTRYPOINT ["/docker-entrypoint-omnicare.sh"]
