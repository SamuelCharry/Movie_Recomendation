import React, { useState } from 'react';

/**
 * RatingControl
 *
 * iOS-style star rating with half-star precision.
 *
 * Props:
 *   value     number | null   current rating (0.5–5.0 or null if unrated)
 *   onChange  function        called with new rating value
 *   readOnly  boolean         display-only mode
 *   size      'sm' | 'md'    default 'md'
 */
function RatingControl({ value, onChange, readOnly = false, size = 'md' }) {
  const [hoverValue, setHoverValue] = useState(null);

  const displayed = hoverValue ?? value ?? 0;
  const starSize  = size === 'sm' ? 17 : 22;
  const numSize   = size === 'sm' ? '0.75rem' : '0.875rem';

  const getStarState = (index) => {
    if (displayed >= index)         return 'full';
    if (displayed >= index - 0.5)   return 'half';
    return 'empty';
  };

  const handleMouseMove = (e, index) => {
    if (readOnly) return;
    const { left, width } = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - left;
    setHoverValue(x < width / 2 ? index - 0.5 : index);
  };

  const handleClick = (e, index) => {
    if (readOnly) return;
    const { left, width } = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - left;
    const newRating = x < width / 2 ? index - 0.5 : index;
    // Clicking the same value deselects
    onChange(value === newRating ? null : newRating);
  };

  return (
    <div
      style={{ display: 'inline-flex', alignItems: 'center', gap: 2 }}
      onMouseLeave={() => !readOnly && setHoverValue(null)}
    >
      {[1, 2, 3, 4, 5].map((index) => {
        const state = getStarState(index);
        return (
          <span
            key={index}
            onMouseMove={(e) => handleMouseMove(e, index)}
            onClick={(e)  => handleClick(e, index)}
            style={{
              fontSize:  starSize,
              lineHeight: 1,
              cursor:    readOnly ? 'default' : 'pointer',
              userSelect: 'none',
              display:   'inline-block',
              position:  'relative',
              color:     state === 'empty' ? '#E5E5EA' : '#FF9500',
              opacity:   state === 'half' ? 0.6 : 1,
              transition: 'color 0.1s, opacity 0.1s',
            }}
            title={!readOnly ? `Rate ${index - 0.5} or ${index}` : undefined}
          >
            {state === 'empty' ? '☆' : '★'}
          </span>
        );
      })}
      <span style={{
        marginLeft: 5,
        fontSize:   numSize,
        color:      displayed > 0 ? '#FF9500' : '#C7C7CC',
        fontVariantNumeric: 'tabular-nums',
        fontWeight: 500,
        minWidth: '2.2ch',
      }}>
        {displayed > 0 ? displayed.toFixed(1) : '—'}
      </span>
    </div>
  );
}

export default RatingControl;
