#MODIFY the following variables (STUDENTID1, STUDENTID2, GROUP) with your infos
STUDENTID1=HERE_ID_OF_THE_STUDENT
STUDENTID2=HERE_ID_OF_THE_STUDENT
GROUP=HERE_GROUP_ID



ASSIGNMENTS=$(filter assignment_%,$(wildcard *))

.PHONY: $(ASSIGNMENTS)
all: $(ASSIGNMENTS)

$(ASSIGNMENTS):
	$(MAKE) -C $@


CFLAGS=-I./include
.PHONY: clean submission demo

demo:
	$(MAKE) -C demo

$(basename $(filter %.c,$(wildcard demo/*)) ): $(filter-out obj/main.o, $(OBJ))
	gcc $(CFLAGS) $@.c $(filter-out obj/main.o, $(OBJ)) -o $@ $(LIBS)

submission_%:
	$(MAKE) -C assignment_$(patsubst submission_%,%,$@) clean
	tar czfh heat_assignment_$(patsubst submission_%,%,$@)_group_$(GROUP)_$(STUDENTID1)_$(STUDENTID2).tar.gz assignment_$(patsubst submission_%,%,$@)

clean:
	-rm -f *.tar.gz
	$(MAKE) -C demo clean
	$(foreach assignment,$(ASSIGNMENTS),$(MAKE) -C $(assignment) clean;) 
