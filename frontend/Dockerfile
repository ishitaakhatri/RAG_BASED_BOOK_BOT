# Stage 1: Build the React Application
FROM node:18-alpine as build

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code and build
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine
# Copy the build output to Nginx's html directory
COPY --from=build /app/dist /usr/share/nginx/html
# We will use a custom Nginx config in the main compose file, 
# so we don't need a complex config here for now.
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]