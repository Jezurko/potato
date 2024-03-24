#pragma once

#include <optional>

namespace potato::util {
    template< typename ...types >
    struct type_list;
    namespace detail {
        using empty_t_list = type_list<>;

        template< typename list >
        struct front;

        template<>
        struct front< empty_t_list >{
            using type = std::nullopt_t;
        };

        template< typename head, typename ...tail >
        struct front< type_list< head, tail... > >{
            using type = head;
        };

        template< typename type_list >
        struct pop_front;

        template< typename head, typename ...tail >
        struct pop_front< type_list< head, tail... > > {
            using type = type_list< tail... >;
        };

        template<>
        struct pop_front< empty_t_list > {
            using type = type_list< empty_t_list >;
        };
    } // namespace detail

    template< typename ...types >
    struct type_list{
        using self = type_list;
        static constexpr std::size_t size = sizeof...(types);
        static constexpr bool empty = size == 0;
        using head = typename detail::front< self >::type;
        using tail = typename detail::pop_front< self >::type;
    };

} // namespace potato::util
