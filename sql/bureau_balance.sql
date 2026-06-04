CREATE TABLE bureau_balance (
    SK_ID_BUREAU    INTEGER,
    MONTHS_BALANCE  INTEGER,
    STATUS          VARCHAR(10)  
);

COPY public.bureau_balance FROM 'C:/Home Credit Default Risk/Home-Credit-Default-Risk/data/bureau_balance.csv' 
WITH (FORMAT csv, HEADER true, DELIMITER ',', ENCODING 'UTF8');