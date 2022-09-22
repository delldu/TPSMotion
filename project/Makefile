all:
	@echo "Help"
	@echo "  make format"
	@echo "  make install"
	@echo "  make uninstall"
	@echo "  make clean"

format:
	@echo
	find image_animation -name "*.py" -exec black -l 120 {} \;

install:
	python setup.py install

uninstall:
	python setup.py uninstall

clean:
	rm -rf build dist image_animation.egg-info
