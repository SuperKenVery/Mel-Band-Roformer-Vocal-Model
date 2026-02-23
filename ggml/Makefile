# Makefile for MelBand RoFormer GGML implementation

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O3 -march=native -fPIC
LDFLAGS = -lm

# GGML settings - adjust these paths based on your GGML installation
GGML_DIR = /usr/local
GGML_INCLUDE = -I$(GGML_DIR)/include
GGML_LIB = -L$(GGML_DIR)/lib -lggml

# If GGML is built locally, use this instead:
# GGML_DIR = ./ggml
# GGML_INCLUDE = -I$(GGML_DIR)/include -I$(GGML_DIR)/src
# GGML_LIB = $(GGML_DIR)/build/src/libggml.a

INCLUDES = $(GGML_INCLUDE)
LIBS = $(GGML_LIB) $(LDFLAGS)

SOURCES = mel_band_roformer.c
HEADERS = mel_band_roformer.h
OBJECTS = $(SOURCES:.c=.o)

TARGET_LIB = libmel_band_roformer.a
TARGET_BIN = mel_band_roformer_inference

.PHONY: all clean install

all: $(TARGET_LIB) $(TARGET_BIN)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET_LIB): $(OBJECTS)
	ar rcs $@ $^

$(TARGET_BIN): main.c $(TARGET_LIB)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(TARGET_LIB) $(LIBS) -o $@

clean:
	rm -f $(OBJECTS) $(TARGET_LIB) $(TARGET_BIN)

install: $(TARGET_LIB) $(TARGET_BIN)
	install -d $(PREFIX)/lib $(PREFIX)/bin $(PREFIX)/include
	install -m 644 $(TARGET_LIB) $(PREFIX)/lib/
	install -m 755 $(TARGET_BIN) $(PREFIX)/bin/
	install -m 644 $(HEADERS) $(PREFIX)/include/

# Help target
help:
	@echo "MelBand RoFormer GGML Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build library and executable (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to PREFIX (default: /usr/local)"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  CC        - C compiler (default: gcc)"
	@echo "  CFLAGS    - Compiler flags"
	@echo "  PREFIX    - Installation prefix (default: /usr/local)"
	@echo "  GGML_DIR  - GGML installation directory"
