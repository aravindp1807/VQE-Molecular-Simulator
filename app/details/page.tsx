'use client';

import { useSearchParams } from 'next/navigation';
import { useState, useEffect, Suspense } from 'react';
import Link from 'next/link';

interface ItemDetails {
  id: string;
  name: string;
  category: string;
  description: string;
  imageUrl: string;
  applications?: string[];
}

function DetailsContent() {
  const searchParams = useSearchParams();
  
  const [details, setDetails] = useState<ItemDetails>({
    id: '',
    name: '',
    category: '',
    description: '',
    imageUrl: '/placeholder.png',
    applications: [],
  });
  
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    setLoading(true);
    
    const id = searchParams.get('id') || '';
    const name = searchParams.get('name') || '';
    const category = searchParams.get('category') || '';
    const description = searchParams.get('description') || '';
    const imageUrl = searchParams.get('imageUrl') || '';
    
    const applications: string[] = [];
    const application1 = searchParams.get('application1');
    const application2 = searchParams.get('application2');
    const application3 = searchParams.get('application3');
    
    if (application1) applications.push(application1);
    if (application2) applications.push(application2);
    if (application3) applications.push(application3);
    
    setDetails({
      id,
      name,
      category,
      description,
      imageUrl,
      applications,
    });
    
    setLoading(false);
  }, [searchParams]);

  return (
    <main className="min-h-screen bg-black text-white p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <Link href="/curated-list" className="text-orange-400 hover:text-orange-300 font-mono transition-colors">
            &larr; Back to Curated List
          </Link>
        </div>

        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-amber-500"></div>
          </div>
        ) : details.name ? (
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-8 shadow-2xl">
            <div className="flex flex-col md:flex-row gap-8 items-start">
              <div className="flex-1">
                <span className="inline-block px-3 py-1 bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-full text-xs font-mono mb-4">
                  {details.category.toUpperCase()}
                </span>
                <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-red-500 bg-clip-text text-transparent">
                  {details.name}
                </h1>
                <p className="text-zinc-300 text-lg leading-relaxed mb-6">
                  {details.description}
                </p>
                
                {details.applications && details.applications.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-amber-400 font-mono mb-2">Key Applications:</h3>
                    <ul className="list-disc list-inside space-y-1 text-zinc-300">
                      {details.applications.map((app, idx) => (
                        <li key={idx}>{app}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
            
            <div className="mt-8 pt-6 border-t border-zinc-800 flex justify-between items-center">
              <Link 
                href="/simulation" 
                className="px-6 py-3 bg-gradient-to-r from-amber-500 to-red-600 text-white rounded-lg font-medium hover:opacity-90 transition-opacity"
              >
                Simulate {details.name} in VQE Core &rarr;
              </Link>
            </div>
          </div>
        ) : (
          <div className="text-center py-16 border border-dashed border-zinc-800 rounded-lg">
            <p className="text-xl text-zinc-400 mb-4">
              No item selected. Please select an item from the curated list.
            </p>
            <Link 
              href="/curated-list" 
              className="px-4 py-2 bg-gradient-to-r from-amber-500 to-red-500 text-white rounded-md font-medium"
            >
              Go to Curated List
            </Link>
          </div>
        )}
      </div>
    </main>
  );
}

export default function DetailsPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-amber-500"></div>
      </div>
    }>
      <DetailsContent />
    </Suspense>
  );
}