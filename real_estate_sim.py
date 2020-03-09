import pandas as pd
from typing import Tuple, List, Sequence, Union

class HousingMarket(object):
    
    def __init__(self, mkt_value=None, monthly_rent=None, growth=0.05, 
                 price_to_rent=None, rental_tax_n_depreciation=0.1, capital_gain=0.2, 
                 transaction_fee=0.05, mortgage_rate=0.04, property_tax=0.015,
                 interest_rate=0.015, mortgage_fee=0.01, downpayment=0.03,
                 management_fee=0.08):
        """
        We prioritize calculating results given mkt_value and monly_rent
        if not provided, default housing value is 500,000
        
        For price_to_rent, values can be None, float, or (float, float)
        - None: look at mkt_value and monly_rent
        - float: assume price_to_rent stays constant
        - (float, float): (start ptr, end prt), and assume linear change
            - use this to model development in the general market. ptr ratio
                tend to increase as the markets mature
        
        - mkt_value, monthly_rent are in absolute dollar value.
        - rental_tax_n_depreciation, capital_gain, transaction_fee, peroperty_tax,
            downpayment, management_fee are all spending in the form of prob.
        - mortgage_rate, interest_rate are probabilities.
            - the former determines monthly payments, and latter determines the
                present value of future cash flows.
        """
        self.mkt_value = mkt_value
        self.monthly_rent = monthly_rent  # initial monthly_rent
        self.growth = growth
        if type(price_to_rent) == tuple:
            self.price_to_rent = price_to_rent 
        elif str(price_to_rent).isnumeric():
            self.price_to_rent = (price_to_rent, price_to_rent)
        else:
            self.price_to_rent = None
        self.capital_gain = capital_gain
        self.transaction_fee = transaction_fee
        self.mortgage_rate = mortgage_rate
        self.property_tax_annual = property_tax
        self.mortgage_fee = mortgage_fee
        self.downpayment = downpayment
        self.management_fee = management_fee 
        self.rental_tax_n_depreciation = rental_tax_n_depreciation
        self.interest_rate = interest_rate
        
        if mkt_value is None and monthly_rent is None and price_to_rent is None:
            print("No values provided. Either provide (mkt_value, monthly_rent) or price_to_rent.")
            return
        
        if (mkt_value is not None and monthly_rent is not None):
            ptr = mkt_value / (12.0 * monthly_rent)
            self.price_to_rent = (ptr, ptr)
        elif price_to_rent is not None:
            self.mkt_value = mkt_value or 5e5  # set default market value
            
        self.capital = self.mkt_value * downpayment

        self.print_args()
    
    def run_simulation(self, years=30):
        """ Run simulation given the number of years 
        
        returns 
        - {"total_ret": annualized_total_ret,
                "rental_ret": annual_rental_ret,
                "appreciation": annualized_appreciation,
                "income_stream": annual_income_stream}
        - df - contain every bit of detail
        """
        self.years = years
        
        # initialize df
        df = pd.DataFrame({"month": range(1, 13)})
        df = pd.concat([df.copy() for _ in range(years)])
        df.reset_index(drop=True, inplace=True)
        df["years"] = df.index // 12 + 1
        
        # insert values
        df["growth"] = (self.growth+1)**(1/12)-1  # monthly growth
        df["cum_growth"] = (df.growth + 1).cumprod()
        df["market_value"] = self.mkt_value * df["cum_growth"]  # updated market value
        df["ptr_ratio"] = self.calc_running_ptr_ratio(df, self.price_to_rent)
        df["monthly_rent"] = df["market_value"] / df["ptr_ratio"] / 12 
        self.pretax_df = df.copy()  # checkpointing
        
        # mortgage payments
        df = self.mortgage_payments(df)
        
        # --- spendings --- 
        df["monthly_spending"] = df["monthly_rent"] * self.management_fee
        df["monthly_spending"] += (df["monthly_rent"] - 
                                  df["monthly_mortgage"]) * self.rental_tax_n_depreciation
        df["monthly_spending"] += df[["month", "market_value"]].apply(self.calc_property_tax, axis=1) 
             
        # net income
        df["net_income"] = df["monthly_rent"] - df["monthly_mortgage"] - df["monthly_spending"]
        results = self.calc_return(df)
        return results, df
    
    def calc_return(self, df):
        """ calculate returns given df """
        r = (1 + self.interest_rate / 12)
        df["discount_factor"] = r ** (df.index + 1)
        df["discounted_net_income"] = df["net_income"] / df["discount_factor"]
        total_ret = df["discounted_net_income"].sum()
        # cap_gain: appreciation
        # total_ret: appreciation + rental income
        cap_gain = (df.loc[df.index[-1], "market_value"]*(1-self.transaction_fee) # after agent fee
                    -self.capital)*(1-self.capital_gain)
        cap_gain /= df.loc[df.index[-1], "discount_factor"]
        total_ret += cap_gain + self.capital
        total_ret /= self.capital
        
        # total return, including appreciation and rent
        annualized_total_ret = total_ret**(1.0/self.years)-1
        
        # appreciation
        annualized_appreciation = (cap_gain/self.capital)**(1.0/self.years)-1
        
        # calculate annual return
        annual_rental_ret = df[["years", "market_value", "net_income"]].groupby("years").apply(self.calc_annual_ret)
        
        # calculate annual income stream
        annual_income_stream = df[["years", "net_income"]].groupby("years").apply(lambda x: x.net_income.sum())
        
        # calculate annual rental return. It's roughly 4% accourding to Randy
        return {"total_ret": annualized_total_ret,
                "rental_ret": annual_rental_ret,
                "appreciation": annualized_appreciation,
                "income_stream": annual_income_stream}
    
    def calc_annual_ret(self, df):
        """ approximate annual return """
        income = df["net_income"].sum()
        mean_mkt_val = df["market_value"].mean()
        return income/mean_mkt_val
    
    def mortgage_payments(self, df):
        """ calculate monthly mortgage payments """
        p = self.mkt_value * (1-self.downpayment) * (1+self.mortgage_fee)
        rate_m = self.mortgage_rate / 12
        n = len(df)
        df["monthly_mortgage"] = p*rate_m*(1+rate_m)**n/((1+rate_m)**n-1)
        return df
    
    def calc_property_tax(self, row):
        """ pay property tax as a part of monthly expense in decenmber """
        if row["month"] == 12:
            return row["market_value"] * self.property_tax_annual
        return 0
    
    def calc_running_ptr_ratio(self, df, ptr: Tuple):
        """ linearly interpolate price to rent ratio """
        # ptr - price to rent ratio 
        assert(len(ptr) == 2)
        start, end = ptr
        df["ptr_ratio_temp_"] = start
        df["ptr_ratio_temp_"] += (end-start)/(len(df)-1)*df.index
        result = df["ptr_ratio_temp_"].copy()
        df.drop("ptr_ratio_temp_", axis=1, inplace=True)
        return result
    
    def print_args(self):
        """ print args as df """
        print("Args provided:")
        args = vars(self).copy()
        for k in args:
            args[k] = str(args[k])
        df = pd.DataFrame(args, index=[0]).T
        df.columns = ["values"] + [str(i) for i in range(len(df.columns)-1)]
        print(df["values"].to_frame())
        