library(stringdist)
library(stringr)
library(dplyr)

get_dist = function(x, y, max_char, method = "osa"){
  x = str_replace(x, "[[:space:]]", "") 
  y = str_replace(y, "[[:space:]]", "") 
  # if(method == 'binary')
  #   outer(x, y, "==")
  # else
    stringdist::stringdist(x, y, method = method) / max_char
}

get_languagedist = function(x){
  # find the longest word in the set
  max_char = sapply(x, nchar) %>% max()
  # get the distance between al pairs of words & standardize by longest word
  m = outer(x, x, get_dist, max_char = max_char) %>% .[lower.tri(.)] %>% c(.)
  # format the output nicely
  new_colnames = outer(names(x), names(x), paste0)
  names(m) = new_colnames[lower.tri(new_colnames)]
  
  m
}

forms = read.csv('./section_5/data/g0_relativeage_forms.csv')

# how many dialects do we have:
glottocodes = str_extract(forms$Language_ID, "[a-z]{4}[0-9]{4}")
#table(glottocodes) %>% sort()

forms$glottocodes = glottocodes
# just take one from each glottocode (even though they are dialects)
forms = forms %>% 
  group_by(glottocodes) %>% 
  slice(1)

#dim(forms)
## remove all IDS languages
# ids = read.csv('private/useable-IDS-languages.csv')
# ids$glottocodes = str_extract(ids$Language_ID, "[a-z]{4}[0-9]{4}")
# remove_ids = ids$glottocodes#[!ids$can_use] ## 
# forms = forms %>% filter(!glottocodes %in% remove_ids)
# dim(forms)

#x = outer(forms[1,2:42], forms[1,2:42], get_dist, max_char = 10)

strdist_matrix = apply(forms[,2:41], 1, get_languagedist) %>% t() %>% as.data.frame(.)
# dim(strdist_matrix)
# colnames(strdist_matrix)

strdist_matrix$Language_ID = forms$Language_ID

write.csv(strdist_matrix, './section_5/processed_data/g0_relativeage_uniqueglottocodes_idsrem_dist.csv', row.names = FALSE)
