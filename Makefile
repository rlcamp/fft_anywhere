# autogenerated
# overrideable vars used in implicit make rules
# only default to -march=native if not on an arm mac
ifeq (,$(findstring arm64,$(shell uname -m)))
TARGET_ARCH ?= -march=native
endif

CFLAGS ?= -O3 -ffinite-math-only -fno-signed-zeros -fno-rounding-math -fno-trapping-math -fno-math-errno
# older versions of gcc need -fcx-limited-range, in others its effect is implied by -ffinite-math-only
ifeq (0,$(shell cc -fcx-limited-range -x c -o /dev/null -c - < /dev/null 2>/dev/null; echo $$?))
	CFLAGS += -fcx-limited-range
endif

CPPFLAGS += -Wall -Wextra -Wshadow -Wmissing-prototypes
LDFLAGS += ${CFLAGS}

# list of targets to build, generated from .c files containing a main() function:

TARGETS=check

all : ${TARGETS}

# for each target, the list of objects to link, generated by recursively crawling include statements with a corresponding .c file:

check : check.o fft_anywhere.o

# for each object, the list of headers it depends on, generated by recursively crawling include statements:

check.o : fft_anywhere.h
fft_anywhere.o : fft_anywhere.h

*.o : Makefile

clean :
	$(RM) -rf *.o *.dSYM ${TARGETS}
.PHONY: clean all
