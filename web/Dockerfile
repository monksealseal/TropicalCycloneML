FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Add custom nginx config optimized for SPA / static site
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static site files
COPY . /usr/share/nginx/html/

# Cloud Run injects PORT env var (default 8080)
# nginx.conf uses envsubst to read it at runtime
ENV PORT=8080

# Use envsubst to inject PORT into nginx config at startup
CMD sh -c "envsubst '\$PORT' < /etc/nginx/conf.d/default.conf > /etc/nginx/conf.d/default.conf.tmp && mv /etc/nginx/conf.d/default.conf.tmp /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"

EXPOSE 8080
