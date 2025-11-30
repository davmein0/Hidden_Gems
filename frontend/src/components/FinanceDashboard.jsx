import React from "react";
import WatchlistItem from "./WatchlistItem";
import IndexCard from "./IndexCard";

export default function FinanceDashboard()
{
    const watchlist = [
        { ticker: "SIXB", sector: "Materials", price: "930.64", change: "-0.58%" },
        { ticker: "SIXC", sector: "Communications", price: "584.50", change: "-1.41%" },
        { ticker: "SIXE", sector: "Energy", price: "951.12", change: "+0.26%" },
        { ticker: "SIXI", sector: "Industrials", price: "1532.44", change: "-1.52%" },
        { ticker: "SIXM", sector: "Financials", price: "652.33", change: "-1.31%" },
        { ticker: "SIXR", sector: "Staples", price: "780.25", change: "+0.03%" },
        { ticker: "SIXRE", sector: "Real Estate", price: "200.22", change: "-1.33%" },
        { ticker: "SIXT", sector: "Technology", price: "2884.73", change: "-2.53%" },
    ];
    
    return (
        <div className="min-h-screen bg-[#0d0f16] text-white flex flex-col">
            {/* HEADER */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
                <div className="text-2xl font-semibold">
                    Google Finance <span className="text-gray-400 text-sm">Beta</span>
                </div>

                <div className="flex items-center gap-3 text-sm text-gray-300">
                    <span>Classic</span>
                    <span className="text-green-400">Beta</span>
                    <span>%</span>
                    <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center">
                        L
                    </div>
                </div>
            </div>

            <div className="flex flex-1">
                {/* LEFT SIDEBAR */}
                <div className="w-[22%] border-r border-gray-700 p-6">
                    <h2 className="text-xl mb-4">WATCHLIST:</h2>

                    <div className="flex flex-col gap-4">
                        {watchlist.map((item, i) => (
                            <WatchlistItem key={i} {...item} />
                        ))}
                    </div>
                </div>

                {/* CENTER GRID */}
                <div className="w-[48%] p-6 overflow-y-auto">
                    <h2 className="text-center text-xl mb-6">
                        LIST OF MIDCAP COMPANIES
                    </h2>

                    <div className="grid grid-cols-3 gap-4">
                        {[...Array(9)].map((_, i) => (
                            <IndexCard
                                key={i}
                                title="S&P 500"
                                value="6,737.49"
                                sub="(-113.43)"
                                change="-1.66%"
                            />
                        ))}
                    </div>
                </div>

                {/* RIGHT PANEL */}
                <div className="w-[30%] p-6 border-l border-gray-700">
                    <div className="flex items-center bg-[#131622] p-3 rounded-xl mb-6">
                        <span className="text-gray-400 mr-3">%</span>
                        <input
                            placeholder="SEARCH"
                            className="flex-1 bg-transparent outline-none text-sm"
                        />
                    </div>

                    <div className="text-gray-500 text-center mt-20 text-lg">
                        DISPLAY OUTPUT FOR RESEARCH
                    </div>
                </div>

            </div>
        </div>
    );
}