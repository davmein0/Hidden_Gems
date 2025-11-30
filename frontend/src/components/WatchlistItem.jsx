import React from "react";
import { COLORS } from "../styles/colors";

export default function WatchlistItem({ ticker, sector, price, change }) {
    const positive = change.trim().startsWith("+");

    return (
        <div className="flex items-center justify-between bg-[#131622] p-3 rounded-xl">
            <div>
                <div className="font-semibold">{ticker}</div>
                <div className="text-xs text-gray-400">{sector}</div>
            </div>
        </div>
    )
}