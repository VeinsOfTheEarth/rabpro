# ---- Explore raw netcdf ----
library(tidync)
library(lubridate)
library(dplyr)
library(ncmeta)

filename <- "C:/Users/jsta/Downloads/ndep-noy_histsoc_monthly_1901_2018.nc"

nc_atts(filename)$value
data.frame(tidync::tidync(filename)$dimension)$length[1] # nlayers
data.frame(ncmeta::nc_inq(filename))
data.frame(nc_vars(filename))

# ---
# months since 1850-01-01 00:00
as.period(as.Date("1901-01-01") - as.Date("1850-01-01"))

# from 1901-01-01 to
month_idx <- seq((51 * 12), ((2018-1901) * 12) + 11 + ((51 * 12)))
length(month_idx)
head(month_idx)
tail(month_idx)

# 1416

(subs <- tidync(filename) %>% hyper_filter(time = time == month_idx[1]))
(subs <- tidync(filename) %>% hyper_filter(time = time == month_idx[length(month_idx)]))

nlayers <- 1405 # length(month_idx)

# ----

year_idx <- seq(from = as.Date("1960-01-01"),
  to = as.Date("2015-01-01"), by = "year") -
  as.Date("1900-01-01")

(subs <- tidync(filename) %>% hyper_filter(time = time == year_idx[2]))
year_idx <- seq(from = as.Date("1960-01-01"),
  to = as.Date("2015-01-01"), by = "year") -
  as.Date("1900-01-01")

(subs <- tidync(filename) %>% hyper_filter(time = time <= 612))

nc_var(filename, 3)


dplyr::filter(nc_atts(filename), name == "calendar")$value