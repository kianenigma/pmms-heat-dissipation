all: $(filter-out Makefile,$(wildcard *))

.PHONY: $(filter-out Makefile,$(wildcard *)) clean $(addprefix clean,$(filter-out Makefile,$(wildcard *)))

$(filter-out Makefile,$(wildcard *)):
	$(MAKE) -C $@

clean: $(addprefix clean_,$(filter-out Makefile,$(wildcard *)))
	@echo $+
	

$(addprefix clean_,$(filter-out Makefile,$(wildcard *))):
	@echo clean $@
	$(MAKE) -C $(subst clean_,,$@) clean
