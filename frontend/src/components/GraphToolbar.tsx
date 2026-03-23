import React, { useState } from 'react';
import { useReactFlow, Panel } from '@xyflow/react';
import { toPng } from 'html-to-image';
import { Search, Download } from 'lucide-react';

export function GraphToolbar() {
    const { getNodes, setNodes, fitView } = useReactFlow();
    const [search, setSearch] = useState('');

    const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = e.target.value.toLowerCase();
        setSearch(val);
        const nodes = getNodes();
        
        let foundNode = null;
        const newNodes = nodes.map(n => {
            const matches = val && typeof n.data.label === 'string' && n.data.label.toLowerCase().includes(val);
            if (matches && !foundNode) foundNode = n;
            
            return {
                ...n,
                style: {
                    ...n.style,
                    opacity: (!val || matches) ? 1 : 0.2,
                    boxShadow: matches ? '0 0 0 3px #ef4444' : (n.style?.boxShadow || 'none')
                }
            };
        });
        
        setNodes(newNodes);
        if (foundNode) {
            fitView({ nodes: [foundNode], duration: 800, maxZoom: 1.2 });
        }
    };

    const handleExport = () => {
        const vp = document.querySelector('.react-flow__viewport') as HTMLElement;
        if (!vp) return;
        toPng(vp, { backgroundColor: '#f8fafc' }).then(dataUrl => {
            const a = document.createElement('a');
            a.setAttribute('download', 'coherence-graph.png');
            a.setAttribute('href', dataUrl);
            a.click();
        });
    };

    return (
        <Panel position="top-right" style={{ 
            display: 'flex', 
            gap: '8px', 
            background: 'var(--bg-surface, #1e1e2e)', 
            padding: '8px', 
            borderRadius: '8px', 
            border: '1px solid var(--border, #333)',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)' 
        }}>
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                <Search size={14} style={{ position: 'absolute', left: '8px', color: '#888' }} />
                <input 
                    type="text" 
                    placeholder="Search nodes..." 
                    value={search} 
                    onChange={handleSearch}
                    style={{ 
                        padding: '4px 8px 4px 28px', 
                        border: '1px solid var(--border, #333)', 
                        borderRadius: '4px', 
                        fontSize: '12px',
                        background: '#0f172a',
                        color: '#fff',
                        outline: 'none',
                        width: '180px'
                    }}
                />
            </div>
            <button 
                onClick={handleExport} 
                style={{ 
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '4px 8px', 
                    background: 'var(--plasma, #00b4ff)', 
                    color: '#fff', 
                    border: 'none', 
                    borderRadius: '4px', 
                    cursor: 'pointer', 
                    fontSize: '12px',
                    fontWeight: 600
                }}>
                <Download size={14} /> PNG
            </button>
        </Panel>
    );
}
