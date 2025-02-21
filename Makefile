.PHONY: build run stop clean test

build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

clean:
	docker-compose down -v
	docker system prune -f

test:
	docker-compose run app python3 test_setup.py 