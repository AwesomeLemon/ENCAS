from .referencebased_subset_selector import ReferenceBasedSubsetSelector

# since only 1 subset selector is used in the paper, this level of indirectness is not needed
# But if I (or someone else) want to add another one later, this would be convenient.
selector_name_to_class = {
    'reference': ReferenceBasedSubsetSelector
}

def create_subset_selector(name, n_to_select, **kwargs):
    clazz = selector_name_to_class[name]
    return clazz(n_to_select, n_objs=2, **kwargs)