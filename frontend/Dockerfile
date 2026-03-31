# Stage 1: Base dependencies
FROM node:20-alpine AS base

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

# Stage 2: Development environment (for HMR)
FROM base AS development
COPY . .
# The command and port are overridden in docker-compose.yaml for dev mode

# Stage 3: Build the React application for production
FROM base AS builder
COPY . .
RUN npm run build

# Stage 4: Serve the application with Nginx (Production)
FROM nginx:alpine AS production

# Copy the build output to replace the default nginx contents
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
