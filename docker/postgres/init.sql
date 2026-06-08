-- docker/postgres/init.sql
-- Виконується ОДИН РАЗ при першому старті контейнера

-- Створити базу "data default" (з пробілом — потрібні лапки)
SELECT 'CREATE DATABASE "data default" OWNER postgres'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'data default'
)\gexec

-- Повідомлення
\echo '✅ Database "data default" ready'